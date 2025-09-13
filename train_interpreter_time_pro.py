import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import sys

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier_time_pro import (
    load_ensemble, compute_iou, compute_dice, 
    predict_labels_with_aggregation,
    MultiTimestepExpertModel
)
from src.datasets_time_pro import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors_time_pro import create_feature_extractor, collect_features, get_required_feature_levels

from guided_diffusion_yandex.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion_yandex.guided_diffusion.dist_util import dev
import numpy as np
import time

 

def prepare_data(args, feature_extractor):
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 
    
    feature_dir = "feature_dir/" + args['class']
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
        compute_features = True
    else:
        compute_features = False

    # Get required feature levels based on configuration
    required_feature_levels = get_required_feature_levels(args['blocks'], args['image_size'])
    print(f"Required feature levels for blocks {args['blocks']}: {sorted(required_feature_levels)}")

    for row, (img, label) in enumerate(tqdm(dataset)):
        if compute_features:
            img = img[None].to(dev())
            features = feature_extractor(img, noise=noise)
            features_by_t = collect_features(args, features)
            
            # Process features for each timestep - only save required levels
            for t, level_features in features_by_t.items():
                # Save only the required feature levels
                for feature_name in required_feature_levels:
                    if feature_name in level_features:
                        feature = level_features[feature_name].cpu().numpy()
                        np.save(os.path.join(feature_dir, f"X_{t.item()}_{feature_name}_{row}.npy"), feature)

        y[row] = label

    return y


def train_unified(args):
    """
    Unified end-to-end training:
    - Train the multi-timestep expert model directly (shared expert + per-timestep aggregation over logits)
    - Save only a single model checkpoint per ensemble member: model_{i}.pth
    - Optional joint mode: update shared expert and aggregation head together end-to-end
    """
    feature_extractor = create_feature_extractor(**args)
    labels = prepare_data(args, feature_extractor)
    required_feature_levels = get_required_feature_levels(args['blocks'], args['image_size'])
    train_data = FeatureDataset(labels, args['class'], args['steps'], required_feature_levels)

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True, num_workers=4)

    for MODEL_NUMBER in range(args['model_num']):
        print(f"Unified training: model {MODEL_NUMBER}")

        model = MultiTimestepExpertModel(num_classes=(args['number_class']), time_steps=args['steps'], aggregation_mode=args.get('aggregation_mode', 'logits'))
        model = model.cuda()
        model = nn.DataParallel(model)

        criterion_ce = nn.CrossEntropyLoss()
        unified_lr = float(args.get('unified_lr', 2e-4))
        shared_lr = unified_lr if args.get('shared_lr') is None else float(args['shared_lr'])

        module_ref = model.module if isinstance(model, nn.DataParallel) else model
        # Match phase-1: initialize expert weights explicitly
        if hasattr(module_ref.shared_model, 'init_weights'):
            module_ref.shared_model.init_weights()
        # Initialize MoE gating conv for stable start
        if hasattr(module_ref, 'init_gating_weights'):
            module_ref.init_gating_weights()
        shared_params = list(module_ref.shared_model.parameters())
        agg_params = [p for n, p in module_ref.named_parameters() if not n.startswith('shared_model.')]

        optimizer_shared = torch.optim.Adam(shared_params, lr=shared_lr)
        optimizer_agg = torch.optim.Adam(agg_params, lr=unified_lr)

        model.train()
        iteration = 0

        unified_epochs = int(args.get('unified_epochs', 100))
        expert_epochs = int(args.get('expert_epochs', unified_epochs))
        aggregation_epochs = int(args.get('aggregation_epochs', unified_epochs))

        # Joint end-to-end training option: update shared expert and MoE together
        if args.get('joint_training', False):
            print(f'Starting Joint end-to-end training for {unified_epochs} epochs')
            optimizer_joint = torch.optim.Adam(
                [
                    {'params': shared_params, 'lr': shared_lr},
                    {'params': agg_params, 'lr': unified_lr},
                ]
            )

            for epoch in range(unified_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for timestep_features, y_batch in train_loader:
                    timestep_features = {t: {k: v.to(dev()) for k, v in features.items()}
                                        for t, features in timestep_features.items()}
                    y_batch_gpu = y_batch.to(dev()).type(torch.long)

                    optimizer_joint.zero_grad()
                    agg_pred = model(timestep_features)
                    loss_joint_ce = criterion_ce(agg_pred, y_batch_gpu)
                    if torch.isnan(loss_joint_ce) or torch.isinf(loss_joint_ce):
                        continue
                    loss_joint_ce.backward()
                    optimizer_joint.step()

                    iteration += 1
                    epoch_loss += float(loss_joint_ce.item())
                    num_batches += 1

                    if iteration % 10 == 0:
                        train_acc = multi_acc(agg_pred, y_batch_gpu)
                        print(f'Unified-Joint - Epoch: {epoch}, Iter: {iteration}, CE: {loss_joint_ce.item():.4f}, Acc: {train_acc:.4f}')

                if num_batches > 0:
                    avg_epoch_loss = epoch_loss / num_batches
                else:
                    avg_epoch_loss = float('inf')
                print(f'Joint Epoch {epoch} average loss: {avg_epoch_loss:.4f}')

            print("Unified joint training completed for current model")
        else:
            # Phase 1: Expert training
            print(f'Starting Expert phase for {expert_epochs} epochs')
            for epoch in range(expert_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for timestep_features, y_batch in train_loader:
                    timestep_features = {t: {k: v.to(dev()) for k, v in features.items()}
                                        for t, features in timestep_features.items()}
                    y_batch_gpu = y_batch.to(dev()).type(torch.long)

                    step_loss_sum_value = 0.0
                    step_count = 0
                    module_ref.shared_model.train()
                    for t in args['steps']:
                        if t not in timestep_features:
                            continue
                        features_t = timestep_features[t]
                        t_tensor = torch.tensor([t]).to(dev())

                        optimizer_shared.zero_grad()
                        pred_t, _ = module_ref.shared_model(features_t, t_tensor)
                        loss_t = criterion_ce(pred_t, y_batch_gpu)
                        if torch.isnan(loss_t) or torch.isinf(loss_t):
                            continue
                        loss_t.backward()
                        optimizer_shared.step()

                        step_loss_sum_value += float(loss_t.item())
                        step_count += 1

                        iteration += 1
                        if iteration % 10 == 0:
                            acc_t = multi_acc(pred_t, y_batch_gpu)
                            print(f'Unified-Expert - Epoch: {epoch}, Iter: {iteration}, Step: {t}, StepCE: {loss_t.item():.4f}, Acc: {acc_t:.4f}')

                    if step_count > 0:
                        loss_step_ce_avg = step_loss_sum_value / step_count
                        epoch_loss += loss_step_ce_avg
                        num_batches += 1

                if num_batches > 0:
                    avg_epoch_loss = epoch_loss / num_batches
                else:
                    avg_epoch_loss = float('inf')
                print(f'Expert Epoch {epoch} average loss: {avg_epoch_loss:.4f}')

            # Phase 2: Aggregation training (freeze expert once)
            for p in module_ref.shared_model.parameters():
                p.requires_grad = False
            module_ref.shared_model.eval()
            print(f'Starting Aggregation phase for {aggregation_epochs} epochs')
            for epoch in range(aggregation_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for timestep_features, y_batch in train_loader:
                    timestep_features = {t: {k: v.to(dev()) for k, v in features.items()}
                                        for t, features in timestep_features.items()}
                    y_batch_gpu = y_batch.to(dev()).type(torch.long)

                    optimizer_agg.zero_grad()
                    agg_pred = model(timestep_features)
                    loss_agg_ce = criterion_ce(agg_pred, y_batch_gpu)
                    if torch.isnan(loss_agg_ce) or torch.isinf(loss_agg_ce):
                        continue
                    loss_agg_ce.backward()
                    optimizer_agg.step()

                    iteration += 1
                    epoch_loss += float(loss_agg_ce.item())
                    num_batches += 1

                    if iteration % 10 == 0:
                        train_acc = multi_acc(agg_pred, y_batch_gpu)
                        print(f'Unified-Aggregation - Epoch: {epoch}, Iter: {iteration}, AggCE: {loss_agg_ce.item():.4f}, Acc: {train_acc:.4f}')

                if num_batches > 0:
                    avg_epoch_loss = epoch_loss / num_batches
                else:
                    avg_epoch_loss = float('inf')
                print(f'Aggregation Epoch {epoch} average loss: {avg_epoch_loss:.4f}')

            print("Unified training completed for current model")

        model_path = os.path.join(args['exp_dir'], f'model_{MODEL_NUMBER}.pth')
        print(f'Saving unified model to: {model_path}')
        torch.save({'model_state_dict': model.state_dict()}, model_path)

def evaluation(args, phase2_models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                          generator=rnd_gen, device=dev())
    else:
        noise = None

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0
    )

    print(f"\n======== Unified models ========")
    print(f"Loaded models count: {len(phase2_models)}")
    for i, model in enumerate(phase2_models):
        if hasattr(model, 'module') and isinstance(model.module, MultiTimestepExpertModel):
            print(f"Model {i}: Unified models")
            print(f"  Timesteps: {model.module.time_steps}")
            if hasattr(model.module, 'moe_gating'):
                print(f"  Gating: MoE (soft), experts={model.module.num_timesteps}")
            else:
                print(f"  Fusion: logit concat + 1x1 conv")
        elif isinstance(model, MultiTimestepExpertModel):
            print(f"Model {i}: Unified models")
            print(f"  Timesteps: {model.time_steps}")
            if hasattr(model, 'moe_gating'):
                print(f"  Gating: MoE (soft), experts={model.num_timesteps}")
            else:
                print(f"  Fusion: logit concat + 1x1 conv")
        else:
            print(f"Model {i}: Other model type")
    print("=====================================\n")

    gts = []
    all_features_by_sample = []

    print("Step 1: Extracting features for all samples...")
    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.to(dev())
            features = feature_extractor(img, noise=noise)
            features_by_t = collect_features(args, features)
            
            features_by_timestep = {}
            for t_tensor, level_features in features_by_t.items():
                t_value = t_tensor.item()
                features_by_timestep[t_value] = {k: v.cpu() for k, v in level_features.items()}
            
            all_features_by_sample.append(features_by_timestep)
            gts.append(label.numpy()[0])
    
    print("\nStep 2: Predicting with unified models...")
    segmentations = []
    mean_probs = []
    
    with torch.no_grad():
        for i, features_by_timestep in enumerate(all_features_by_sample):
            agg_pred, agg_mean_seg = predict_labels_with_aggregation(
                phase2_models, 
                features_by_timestep, 
                args['dim'][:-1]
            )
            
            if isinstance(agg_pred, torch.Tensor):
                agg_pred = agg_pred.cpu().numpy()
             
            segmentations.append(agg_pred)
            mean_probs.append(agg_mean_seg)
    
    

    print('\nEvaluating unified model:')
    if segmentations:
        mean_iou, _ = compute_iou(args, segmentations, gts)
        mean_dice, _ = compute_dice(args, segmentations, gts)
        print(f'Aggregation model - IOU: {mean_iou:.4f}, DICE: {mean_dice:.4f}')
    else:
        print("Warning: No segmentation results generated!")
    return segmentations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int, default=0)
    # Two-phase args removed. Unified training only.
    
    parser.add_argument('--evaluation_only', action='store_true', help='Only evaluate existing models without training')
    parser.add_argument('--unified_epochs', type=int, default=100, help='Number of epochs for unified training')
    parser.add_argument('--unified_lr', type=float, default=2e-3, help='Learning rate for unified training (aggregation head)')
    parser.add_argument('--shared_lr', type=float, default=None, help='Learning rate for shared expert updates (default: unified_lr)')
    parser.add_argument('--aggregation_mode', type=str, default='logits', choices=['logits', 'features'], help='Aggregation mode for multi-timestep model')
    parser.add_argument('--joint_training', action='store_true', help='End-to-end joint training: update shared expert and MoE together')
    
    # Two-phase LR args removed
    # Load the local setting
    args = parser.parse_args()
    
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    
    

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if opts['evaluation_only'] or all(pretrained):
        print('Loading models...')
        phase2_models = load_ensemble(opts, device='cuda')
        
        results = evaluation(opts, phase2_models)
        
        if opts['evaluation_only']:
            print('Evaluation completed, exiting program')
            sys.exit(0)
    if not all(pretrained):
        print('Running unified training...')
        train_unified(opts)
        print('Training completed, loading trained models for evaluation...')
        phase2_models = load_ensemble(opts, device='cuda')
        evaluation(opts, phase2_models)