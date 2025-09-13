import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

import functools
import torch.nn.functional as F
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image
from medpy.metric import binary
from timm.layers import DropPath
from biggans_layers import ccbn, one_hot

from FreqFusion import FreqFusion
class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps,
                which_conv=None, activation=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.time_steps = time_steps
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.bn1 = ccbn(in_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn')
        self.bn2 = ccbn(out_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn')
        self.dropout = nn.Dropout2d(0.2)
        self.drop_path = DropPath(0.2)

    def forward(self, x, t):
        if isinstance(t, torch.Tensor) and t.numel() == 1:
            t_value = t.item()
        else:
            t_value = t[0].item() if isinstance(t, torch.Tensor) else t
        
        if hasattr(self, 'time_steps') and isinstance(self.time_steps, (list, tuple)):
            if t_value in self.time_steps:
                t_idx = self.time_steps.index(t_value)
            else:
                t_idx = 0
        else:
            t_idx = t_value if isinstance(t_value, int) else 0
            
        t_cond = one_hot(torch.tensor([t_idx]).to(x.device), len(self.time_steps))
        
        h = self.activation(self.bn1(x, t_cond))
        h = self.conv1(h)
        h = self.activation(self.bn2(h, t_cond))
        h = self.conv2(h)
        h = self.drop_path(h) + x
        return h

class BigdatasetGANModel(nn.Module):
    def __init__(self, out_dim, time_steps, fusion_type='add'):
        """
        Initialize model
        
        Args:
            out_dim: Number of output channels
            time_steps: List of timesteps
            fusion_type: Feature fusion method, options: 'concat', 'concat_conv', 'add'
        """
        super(BigdatasetGANModel, self).__init__()

        low_feature_channel = 384
        mid_feature_channel = 384
        high_feature_channel = 384
        
        # Input channel dimensions for feature convolutions
        low_input_channels = 2048
        mid_input_channels = 512
        high_input_channels = 256
        
        self.time_steps = time_steps
        self.num_timesteps = len(time_steps) if isinstance(time_steps, (list, tuple)) else 1
        self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)
        self.fusion_type = fusion_type
        
        
        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(low_input_channels, low_input_channels, kernel_size=3, padding=1, groups=low_input_channels),  # Depth-wise conv
            ccbn(low_input_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(low_input_channels, low_feature_channel, kernel_size=1, bias=False),
        )
        
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d(mid_input_channels, mid_input_channels, kernel_size=3, padding=1, groups=mid_input_channels),  # Depth-wise conv
            ccbn(mid_input_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(mid_input_channels, mid_feature_channel, kernel_size=1, bias=False),
        )
        
        if fusion_type == 'concat':
            mid_channels = low_feature_channel + mid_feature_channel
            high_channels = low_feature_channel + mid_feature_channel + high_feature_channel
            final_channels = high_channels
        elif fusion_type == 'concat_conv':
            mid_channels = mid_feature_channel
            high_channels = high_feature_channel
            final_channels = high_feature_channel
            
            self.mid_reduce_conv = nn.Sequential(
                nn.Conv2d(low_feature_channel + mid_feature_channel, mid_feature_channel, kernel_size=1),
                ccbn(mid_feature_channel, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
                nn.ReLU(inplace=False)
            )
            
            self.high_reduce_conv = nn.Sequential(
                nn.Conv2d(mid_feature_channel + high_feature_channel, high_feature_channel, kernel_size=1),
                ccbn(high_feature_channel, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
                nn.ReLU(inplace=False)
            )
        else:  # 'add'
            mid_channels = mid_feature_channel
            high_channels = high_feature_channel
            final_channels = high_feature_channel
        
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=mid_channels,
                                out_channels=mid_channels,
                                which_conv=self.which_conv,
                                activation=self.activation,
                                time_steps=time_steps
                            )
                            
        self.high_feature_conv = nn.Sequential(
            nn.Conv2d(high_input_channels, high_input_channels, kernel_size=3, padding=1, groups=high_input_channels),  # Depth-wise conv
            ccbn(high_input_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(high_input_channels, high_feature_channel, kernel_size=1, bias=False),
        )

        self.high_feature_mix_conv = SegBlock(
                                in_channels=high_channels,
                                out_channels=high_channels,
                                which_conv=self.which_conv,
                                activation=self.activation,
                                time_steps=time_steps
                            )

        self.out_layer = nn.Sequential(
                                ccbn(final_channels, len(time_steps), nn.Linear, eps=1e-4, norm_style='bn'),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(final_channels, out_dim, kernel_size=1)
        )
        
        self.freq_fusion_mid = FreqFusion(
            hr_channels=mid_feature_channel, 
            lr_channels=low_feature_channel,
            time_steps=time_steps
        )
        
        if fusion_type == 'concat':
            self.freq_fusion_high = FreqFusion(
                hr_channels=high_feature_channel, 
                lr_channels=low_feature_channel+mid_feature_channel,
                time_steps=time_steps
            )
        else:
            self.freq_fusion_high = FreqFusion(
                hr_channels=high_feature_channel, 
                lr_channels=mid_feature_channel,
                time_steps=time_steps
            )

    def forward(self, features_dict, t, return_features=False):
        deep_supervision_outputs = []  # Keep for compatibility, but no longer used

        if isinstance(t, torch.Tensor) and t.numel() == 1:
            t_value = t.item()
        else:
            t_value = t[0].item() if isinstance(t, torch.Tensor) else t
            
        if t_value in self.time_steps:
            t_idx = self.time_steps.index(t_value)
        else:
            t_idx = 0
            
        t_cond = one_hot(torch.tensor([t_idx]).to(features_dict['third'].device), len(self.time_steps))
            
        # Process low features
        low_feat = features_dict['third']
        for layer in self.low_feature_conv:
            if isinstance(layer, ccbn):
                low_feat = layer(low_feat, t_cond)
            else:
                low_feat = layer(low_feat)
        
        # Process mid features
        mid_feat = features_dict['fine']
        for layer in self.mid_feature_conv:
            if isinstance(layer, ccbn):
                mid_feat = layer(mid_feat, t_cond)
            else:
                mid_feat = layer(mid_feat)
        
        _, mid_feat, low_feat_up = self.freq_fusion_mid(hr_feat=mid_feat, lr_feat=low_feat, t=t)
        
        if self.fusion_type == 'concat':
            mid_feat = torch.cat([low_feat_up, mid_feat], dim=1)
        elif self.fusion_type == 'concat_conv':
            mid_feat = torch.cat([low_feat_up, mid_feat], dim=1)
            for layer in self.mid_reduce_conv:
                if isinstance(layer, ccbn):
                    mid_feat = layer(mid_feat, t_cond)
                else:
                    mid_feat = layer(mid_feat)
        else:  # 'add'
            mid_feat = mid_feat + low_feat_up
            
        mid_feat = self.mid_feature_mix_conv(mid_feat, t)
        
        # Process high features
        high_feat = features_dict['low']
        for layer in self.high_feature_conv:
            if isinstance(layer, ccbn):
                high_feat = layer(high_feat, t_cond)
            else:
                high_feat = layer(high_feat)
        
        _, high_feat, mid_feat_up = self.freq_fusion_high(hr_feat=high_feat, lr_feat=mid_feat, t=t)
        
        if self.fusion_type == 'concat':
            high_feat = torch.cat([mid_feat_up, high_feat], dim=1)
        elif self.fusion_type == 'concat_conv':
            high_feat = torch.cat([mid_feat_up, high_feat], dim=1)
            for layer in self.high_reduce_conv:
                if isinstance(layer, ccbn):
                    high_feat = layer(high_feat, t_cond)
                else:
                    high_feat = layer(high_feat)
        else:  # 'add'
            high_feat = high_feat + mid_feat_up
            
        high_feat = self.high_feature_mix_conv(high_feat, t)
        
        high_feat = F.interpolate(high_feat, size=256, mode='bilinear', align_corners=False)
        
        if return_features:
            return high_feat
            
        pred = high_feat
        for layer in self.out_layer:
            if isinstance(layer, ccbn):
                pred = layer(pred, t_cond)
            else:
                pred = layer(pred)
        
        return pred, deep_supervision_outputs

class pixel_classifier(nn.Module):
    def __init__(self, num_classes, time_steps):
        super(pixel_classifier, self).__init__()
        self.layers = BigdatasetGANModel(out_dim=num_classes, time_steps=time_steps)
        
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, t, return_features=False):
        return self.layers(x, t, return_features)
        

def predict_labels_with_aggregation(models, features_by_timestep, size):
    """
    Predict segmentation results using timestep feature aggregation
    
    Args:
        models: List of models
        features_by_timestep: Dict with timestep as key and features as value
        size: Output image size
    """
    mean_seg = None
    all_seg = []
    # removed entropy accumulation in unified flow
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            model = models[MODEL_NUMBER]
            
            try:
                is_multi_expert = isinstance(model, MultiTimestepExpertModel)
                is_dataparallel = isinstance(model, nn.DataParallel)
                is_dataparallel_multi_expert = is_dataparallel and isinstance(model.module, MultiTimestepExpertModel)
                
                if is_multi_expert or is_dataparallel_multi_expert:
                    cuda_features = {t: {k: v.cuda() for k, v in features.items()} 
                                    for t, features in features_by_timestep.items()}
                    
                    if is_multi_expert:
                        preds = model(cuda_features)
                    else:
                        preds = model.module(cuda_features)
                        

                else:
                    if is_dataparallel:
                        num_classes = model.module.layers.out_layer[-1].out_channels
                        time_steps = model.module.layers.time_steps
                    else:
                        num_classes = model.layers.out_layer[-1].out_channels
                        time_steps = model.layers.time_steps
                    
                    temp_model = MultiTimestepExpertModel(num_classes=num_classes, time_steps=time_steps)
                    temp_model.shared_model = model
                    temp_model = temp_model.to('cuda').eval()
                    
                    cuda_features = {t: {k: v.cuda() for k, v in features.items()} 
                                    for t, features in features_by_timestep.items()}
                    
                    preds = temp_model(cuda_features)
                
                
                if preds is None:
                    print(f"Warning: Model {MODEL_NUMBER} generated no predictions, skipping")
                    continue
                
                # entropy logging removed in unified flow
                all_seg.append(preds)

                if mean_seg is None:
                    mean_seg = softmax_f(preds)
                else:
                    mean_seg += softmax_f(preds)

                img_seg = oht_to_scalar(preds)
                img_seg = img_seg.reshape(*size)
                img_seg = img_seg.cpu().detach()

                seg_mode_ensemble.append(img_seg)
                
            except Exception as e:
                print(f"Error: Model {MODEL_NUMBER} prediction failed: {e}")
                continue

        if all_seg:
            mean_seg = mean_seg / len(all_seg)
        else:
            print("Warning: No successful model predictions")
            return None, None
    
    if seg_mode_ensemble:
        ensemble_output = torch.stack([torch.from_numpy(img_seg.numpy()) for img_seg in seg_mode_ensemble], dim=0)
        mode_pred, _ = torch.mode(ensemble_output, dim=0)
        

        
        return mode_pred.numpy(), mean_seg
        
    return None, None

def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.jpg')
        )


    
from sklearn.metrics import auc, precision_recall_curve

def compute_auc_pr(args, mean_segs, gts):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])
    
    average_aucs = []

    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue

        all_probs_tmp = []
        all_gts_tmp = []

        for prob, gt in zip(mean_segs, gts):
            if target_num == 0:
                probs_tmp = prob[0][0].reshape(-1)
            else:
                probs_tmp = prob[0][1].reshape(-1)

            gts_tmp = (gt == target_num).astype(int).reshape(-1)

            all_probs_tmp.extend(probs_tmp.tolist())
            all_gts_tmp.extend(gts_tmp.tolist())

        precision, recall, _ = precision_recall_curve(all_gts_tmp, all_probs_tmp)
        auc_value = auc(recall, precision)
        average_aucs.append(auc_value)
        
        print(f"AUC-PR for {class_names[target_num]}: {auc_value:.4f}")

    return np.array(average_aucs).mean(), auc_value

from sklearn.metrics import roc_curve

def compute_auc_roc(args, mean_segs, gts):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])
    
    average_aucs = []

    for target_num in ids:
        if target_num == args['ignore_label']:
            continue

        all_probs_tmp = []
        all_gts_tmp = []

        for prob, gt in zip(mean_segs, gts):
            if target_num == 0:
                probs_tmp = prob[0][0].reshape(-1)
            else:
                probs_tmp = prob[0][1].reshape(-1)

            gts_tmp = (gt == target_num).astype(int).reshape(-1)

            all_probs_tmp.extend(probs_tmp.tolist())
            all_gts_tmp.extend(gts_tmp.tolist())

        fpr, tpr, _ = roc_curve(all_gts_tmp, all_probs_tmp)
        auc_value = auc(fpr, tpr)
        average_aucs.append(auc_value)

        print(f"AUC-ROC for {class_names[target_num]}: {auc_value:.4f}")

    return np.array(average_aucs).mean(), auc_value

def compute_iou(args, preds, gts, print_per_class_ious=False):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean(), iou

def compute_dice(args, preds, gts, print_per_class_dices=False, print_target1_per_sample_dices=False):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])

    intersections = Counter()
    pred_areas = Counter()
    gt_areas = Counter()

    target1_per_sample_dices = []

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        sample_intersections = Counter()
        sample_pred_areas = Counter()
        sample_gt_areas = Counter()

        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue

            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            pred_areas[target_num] += preds_tmp.sum()
            gt_areas[target_num] += gts_tmp.sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()

            sample_pred_areas[target_num] = preds_tmp.sum()
            sample_gt_areas[target_num] = gts_tmp.sum()
            sample_intersections[target_num] = (preds_tmp & gts_tmp).sum()

        if print_target1_per_sample_dices and 1 in ids:
            dice = 2 * sample_intersections[1] / (1e-8 + sample_pred_areas[1] + sample_gt_areas[1])
            target1_per_sample_dices.append(dice)
            print(f"Sample {i} Dice for target_num=1: {dice:.4f}")

    dices = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        dice = 2 * intersections[target_num] / (1e-8 + pred_areas[target_num] + gt_areas[target_num])
        dices.append(dice)
        if print_per_class_dices:
            print(f"DICE for {class_names[target_num]}: {dice:.4f}")

    return np.array(dices).mean(), dice

def compute_accuracy(args, preds, gts, print_per_class_accuracies=True):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])

    correct_predictions = Counter()
    total_pixels = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']:
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)

            correct_predictions[target_num] += (preds_tmp & gts_tmp).sum()
            total_pixels[target_num] += gts_tmp.sum()

    accuracies = []
    for target_num in ids:
        if target_num == args['ignore_label']:
            continue
        accuracy = correct_predictions[target_num] / (1e-8 + total_pixels[target_num])
        accuracies.append(accuracy)
        if print_per_class_accuracies:
            print(f"Accuracy for {class_names[target_num]} {accuracy:.4}")

    return np.array(accuracies).mean(), accuracy
    
def compute_f1_score(args, preds, gts, print_per_class_f1=True):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])

    tp = Counter()  # True Positives
    fp = Counter()  # False Positives
    fn = Counter()  # False Negatives

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']:
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)

            tp[target_num] += (preds_tmp & gts_tmp).sum()
            fp[target_num] += ((preds_tmp == 1) & (gts_tmp == 0)).sum()
            fn[target_num] += ((preds_tmp == 0) & (gts_tmp == 1)).sum()

    f1_scores = []
    for target_num in ids:
        if target_num == args['ignore_label']:
            continue
        precision = tp[target_num] / (1e-8 + tp[target_num] + fp[target_num])
        recall = tp[target_num] / (1e-8 + tp[target_num] + fn[target_num])
        f1_score = 2 * precision * recall / (1e-8 + precision + recall)
        f1_scores.append(f1_score)
        if print_per_class_f1:
            print(f"F1-score for {class_names[target_num]} {f1_score:.4}")

    return np.array(f1_scores).mean()

def compute_hd95(args, preds, gts, print_per_class_hd95=True):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])

    hd95_values = []

    for target_num in ids:
        if target_num == args['ignore_label']:
            continue
        temp_hd95 = []
        for pred, gt in zip(preds, gts):
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            hd95 = binary.hd95(preds_tmp, gts_tmp)
            temp_hd95.append(hd95)
        mean_hd95 = np.array(temp_hd95).mean()
        hd95_values.append(mean_hd95)
        if print_per_class_hd95:
            print(f"HD95 for {class_names[target_num]} {mean_hd95:.4}")
    return np.array(hd95_values).mean()
    
def compute_assd(args, preds, gts, print_per_class_assd=True):
    class_names = get_class_names(args['category'])
    ids = range(args['number_class'])

    assd_values = []

    for target_num in ids:
        if target_num == args['ignore_label']:
            continue
        temp_assd = []
        for pred, gt in zip(preds, gts):
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            assd = binary.assd(preds_tmp, gts_tmp)
            temp_assd.append(assd)
        mean_assd = np.array(temp_assd).mean()
        assd_values.append(mean_assd)
        if print_per_class_assd:
            print(f"ASSD for {class_names[target_num]} {mean_assd:.4}")
    return np.array(assd_values).mean()

def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
            
        try:
            state_dict = torch.load(model_path)['model_state_dict']
        except Exception as e:
            print(f"Warning: Unable to load model file: {model_path}, error: {e}")
            continue
        
        # Detect unified model by presence of shared_model or aggregation head
        is_multi_expert = any(
            k.startswith('shared_model.') or k.startswith('module.shared_model.') or
            ('moe_gating' in k)
            for k in state_dict.keys()
        )
        
        if is_multi_expert:
            model = MultiTimestepExpertModel(num_classes=args["number_class"], time_steps=args['steps'], aggregation_mode=args.get('aggregation_mode', 'logits'))
            
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            
            try:
                model.load_state_dict(state_dict)
                model = model.to(device)
                models.append(model.eval())
            except Exception as e:
                print(f"Model {i}: Failed to load multi-timestep model weights: {e}")
        else:
            model = pixel_classifier(args["number_class"], args['steps'])
            
            if any(key.startswith('module.') for key in state_dict.keys()):
                model = nn.DataParallel(model)
                try:
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    models.append(model.eval())
                except Exception as e:
                    print(f"Model {i}: Failed to load DataParallel model weights: {e}")
            else:
                try:
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    models.append(model.eval())
                except Exception as e:
                    print(f"Model {i}: Failed to load model weights: {e}")
                    
                    try:
                        model = nn.DataParallel(model)
                        model.module.load_state_dict(state_dict)
                        model = model.to(device)
                        models.append(model.eval())
                    except Exception as e2:
                        print(f"Model {i}: All loading attempts failed: {e2}")
        
    if not models:
        print("Warning: No models loaded successfully!")
    else:
        print(f"Successfully loaded {len(models)} models")
        
    return models

class MultiTimestepExpertModel(nn.Module):
    """
    Multi-timestep expert model using a single conditional model to handle all timesteps,
    with per-timestep aggregation over logits or features.
    """
    def __init__(self, num_classes, time_steps, aggregation_mode: str = 'logits'):
        super().__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.num_timesteps = len(time_steps)
        self.aggregation_mode = aggregation_mode

        self.shared_model = pixel_classifier(num_classes=num_classes, time_steps=time_steps)
        
        # Build aggregation head according to aggregation mode
        if self.aggregation_mode == 'logits':
            # Aggregation weights over per-timestep logits
            self.moe_gating = nn.Conv2d(num_classes * self.num_timesteps, self.num_timesteps, kernel_size=1)
        elif self.aggregation_mode == 'features':
            # Determine feature channel dimension from the shared model's head input
            # BigdatasetGANModel.out_layer[-1] is Conv2d(final_channels -> num_classes)
            feature_channels = self.shared_model.layers.out_layer[-1].in_channels
            self.feature_channels = feature_channels
            # Aggregation weights on concatenated high-level features across timesteps
            self.moe_gating_features = nn.Conv2d(self.feature_channels * self.num_timesteps, self.num_timesteps, kernel_size=1)
            # Non-conditional head to map aggregated features to logits
            self.aggregation_out_layer = nn.Sequential(
                nn.BatchNorm2d(self.feature_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.feature_channels, num_classes, kernel_size=1)
            )
        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}. Expected 'logits' or 'features'.")

        self.gating_temperature = 1.0

    def init_gating_weights(self, init_type: str = 'normal', gain: float = 0.02):
        """
        Initialize parameters of the aggregation head.
        Only affects aggregation layers; does not touch `shared_model`.
        """
        def _init_layer(module: nn.Module):
            if isinstance(module, nn.Conv2d):
                if init_type == 'normal':
                    nn.init.normal_(module.weight, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight, gain=gain)
                else:
                    nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Collect and initialize modules
        if hasattr(self, 'moe_gating') and isinstance(self.moe_gating, nn.Conv2d):
            _init_layer(self.moe_gating)
        if hasattr(self, 'moe_gating_features') and isinstance(self.moe_gating_features, nn.Conv2d):
            _init_layer(self.moe_gating_features)
        if hasattr(self, 'aggregation_out_layer') and isinstance(self.aggregation_out_layer, nn.Sequential):
            for m in self.aggregation_out_layer.modules():
                _init_layer(m)

    def forward(self, timestep_features):
        """
        Forward pass that processes features from multiple timesteps and aggregates them at logit level.
        
        Args:
            timestep_features: Dict containing features for each timestep {timestep: features}
            
        Returns:
            Aggregated logits
        """
        if not isinstance(timestep_features, dict):
            print(f"Error: Input should be dict, but received {type(timestep_features)}")
            return None
        
        timestep_features = {int(t) if isinstance(t, torch.Tensor) else t: feat 
                             for t, feat in timestep_features.items()}
        
        missing_steps = [t for t in self.time_steps if t not in timestep_features]
        if missing_steps:
            raise RuntimeError(f"Missing timestep features for steps: {missing_steps}; available: {list(timestep_features.keys())}")
        
        if self.aggregation_mode == 'logits':
            # Compute per-timestep logits using shared expert
            logits_per_timestep = {}
            expert_requires_grad = any(p.requires_grad for p in self.shared_model.parameters())
            for t in self.time_steps:
                features = timestep_features[t]
                t_tensor = torch.tensor([t]).to(features[list(features.keys())[0]].device)
                # Avoid building graphs through experts when they are frozen during aggregation training
                with torch.set_grad_enabled(expert_requires_grad):
                    if isinstance(self.shared_model, nn.DataParallel):
                        logits_t, _ = self.shared_model.module(features, t_tensor)
                    else:
                        logits_t, _ = self.shared_model(features, t_tensor)
                logits_per_timestep[t] = logits_t
            
            # Ensure fixed ordering; missing steps raise earlier
            logits_list_ordered = [logits_per_timestep[t] for t in self.time_steps]
            
            logits_cat = torch.cat(logits_list_ordered, dim=1)  # [B, K*C, H, W]
            gate_scores = self.moe_gating(logits_cat)  # [B, K, H, W]
            gate_weights = torch.softmax(gate_scores / self.gating_temperature, dim=1)

            
            # Stack logits and apply aggregation weights
            logits_stack = torch.stack(logits_list_ordered, dim=0)  # [K, B, C, H, W]
            weights_stack = gate_weights.permute(1, 0, 2, 3).unsqueeze(2)  # [K, B, 1, H, W]
            mixed_logits = torch.sum(logits_stack * weights_stack, dim=0)  # [B, C, H, W]
            return mixed_logits
        else:
            # Feature-level aggregation: use high-level features from shared expert
            features_per_timestep = {}
            expert_requires_grad = any(p.requires_grad for p in self.shared_model.parameters())
            for t in self.time_steps:
                features = timestep_features[t]
                t_tensor = torch.tensor([t]).to(features[list(features.keys())[0]].device)
                # Avoid building graphs through experts when they are frozen during aggregation training
                with torch.set_grad_enabled(expert_requires_grad):
                    if isinstance(self.shared_model, nn.DataParallel):
                        high_feat = self.shared_model.module(features, t_tensor, return_features=True)
                    else:
                        high_feat = self.shared_model(features, t_tensor, return_features=True)
                features_per_timestep[t] = high_feat

            # Ensure fixed ordering
            feature_list_ordered = [features_per_timestep[t] for t in self.time_steps]  # [B,Cf,H,W] each
            features_cat = torch.cat(feature_list_ordered, dim=1)  # [B, K*Cf, H, W]
            gate_scores = self.moe_gating_features(features_cat)  # [B, K, H, W]
            gate_weights = torch.softmax(gate_scores / self.gating_temperature, dim=1)

            

            # Weighted sum of features
            features_stack = torch.stack(feature_list_ordered, dim=0)  # [K, B, Cf, H, W]
            weights_stack = gate_weights.permute(1, 0, 2, 3).unsqueeze(2)  # [K, B, 1, H, W]
            mixed_features = torch.sum(features_stack * weights_stack, dim=0)  # [B, Cf, H, W]

            # Final mapping to logits (time-agnostic)
            logits = self.aggregation_out_layer(mixed_features)
            return logits
        
