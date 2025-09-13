import sys
import torch
from torch import nn
from typing import List, Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    elif model_type == 'mae':
        print("Creating MAE Feature Extractor...")
        feature_extractor = FeatureExtractorMAE(**kwargs)
    elif model_type == 'swav':
        print("Creating SwAV Feature Extractor...")
        feature_extractor = FeatureExtractorSwAV(**kwargs)
    elif model_type == 'swav_w2':
        print("Creating SwAVw2 Feature Extractor...")
        feature_extractor = FeatureExtractorSwAVw2(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    """
    Wrapper to extract features from pretrained DDPMs.
            
    Args:
        steps: List of diffusion steps t
        blocks: List of UNet decoder blocks (currently extracts all blocks for flexibility)
    """
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations for ALL blocks to ensure complete feature availability
        # This allows prepare_features to work with any block configuration
        for idx, block in enumerate(self.model.output_blocks):
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        import guided_diffusion_yandex.guided_diffusion.dist_util as dist_util
        from guided_diffusion_yandex.guided_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        self.model.to(dist_util.dev())
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        import guided_diffusion_yandex.guided_diffusion.dist_util as dist_util
        activations_dict = {}
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            with torch.no_grad():
                model_kwargs = {}
                self.diffusion.ddim_reverse_sample(self.model, x, t)

            # Extract activations
            current_activations = []
            for block in self.feature_blocks:
                current_activations.append(block.activations)
                block.activations = None
            activations_dict[t] = current_activations

        # Per-layer list of activations [N, C, H, W]
        return activations_dict


class FeatureExtractorMAE(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained MAE
    '''
    def __init__(self, num_blocks=12, **kwargs):
        super().__init__(**kwargs)

        # Save features from deep encoder blocks 
        for layer in self.model.blocks[-num_blocks:]:
            layer.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer)

    def _load_pretrained_model(self, model_path, **kwargs):
        import mae
        from functools import partial
        sys.path.append(mae.__path__[0])
        from mae.models_mae import MaskedAutoencoderViT

        # Create MAE with ViT-L-8 backbone 
        model = MaskedAutoencoderViT(
            img_size=256, patch_size=8, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True
        )

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        self.model = model.eval().to(device)

    @torch.no_grad()
    def forward(self, x, **kwargs):
        _, _, ids_restore = self.model.forward_encoder(x, mask_ratio=0)
        ids_restore = ids_restore.unsqueeze(-1)
        sqrt_num_patches = int(self.model.patch_embed.num_patches ** 0.5)
        activations = []
        for block in self.feature_blocks:
            # remove cls token 
            a = block.activations[:, 1:]
            # unshuffle patches
            a = torch.gather(a, dim=1, index=ids_restore.repeat(1, 1, a.shape[2])) 
            # reshape to obtain spatial feature maps
            a = a.permute(0, 2, 1)
            a = a.view(*a.shape[:2], sqrt_num_patches, sqrt_num_patches)

            activations.append(a)
            block.activations = None
        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorSwAV(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained SwAVs 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layers = [self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]

        # Save features from sublayers
        for layer in layers:
            for l in layer[::2]:
                l.register_forward_hook(self.save_hook)
                self.feature_blocks.append(l)

    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50

        model = resnet50(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False) 
        self.model = model.module.eval()

    @torch.no_grad()
    def forward(self, x, **kwargs):
        self.model(x)

        activations = []
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations
    

class FeatureExtractorSwAVw2(FeatureExtractorSwAV):
    ''' 
    Wrapper to extract features from twice wider pretrained SwAVs 
    '''
    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50w2

        model = resnet50w2(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False) 
        self.model = model.module.eval()


def get_required_feature_levels(required_indices, resolution=512):
    """
    Get the actual feature levels that need to be saved/loaded based on required indices.
    
    Args:
        required_indices: List of feature indices actually needed (from config blocks)
        resolution: Image resolution (256 or 512) to determine feature configuration
    
    Returns:
        Set of feature level names that are actually used
    """
    # Feature configuration based on resolution
    if resolution == 256:
        default_config = {
            'first': [0, 1],                    # Early low-resolution features
            'second': [2, 3, 4],               # Early-mid resolution features  
            'third': [5, 6, 7],                # Mid-level features
            'fine': [8, 9, 10],                # Fine-grained features
            'low': [11, 12, 13],               # Low-level semantic features
            'mid': [14, 15, 16],               # Mid-level semantic features
            'high': [17]                       # High-level semantic features (256 resolution)
        }
    else:  # resolution == 512
        default_config = {
            'first': [0, 1],                    # Early low-resolution features
            'second': [2, 3, 4],               # Early-mid resolution features  
            'third': [5, 6, 7],                # Mid-level features
            'fine': [8, 9, 10],                # Fine-grained features
            'low': [11, 12, 13],               # Low-level semantic features
            'mid': [14, 15, 16],               # Mid-level semantic features
            'high': [17, 18, 19, 20]           # High-level semantic features (512 resolution)
        }
    
    # Find which feature levels actually use the required indices
    used_levels = set()
    for level_name, level_indices in default_config.items():
        # Check if this level uses any of the required indices
        if any(idx in required_indices for idx in level_indices):
            used_levels.add(level_name)
    
    return used_levels


def prepare_features(features, required_indices, resolution=512):
    """
    Prepare multi-level features from extracted activations based on hierarchical grouping.
    Automatically determines simplified configuration based on required feature indices.
    
    Args:
        features: List of feature tensors from different network layers
        required_indices: List of feature indices actually needed (from config blocks)
        resolution: Image resolution (256 or 512) to determine feature configuration
    
    Returns:
        Dictionary containing organized features at different hierarchical levels
    """
    
    # Feature configuration based on resolution
    if resolution == 256:
        default_config = {
            'first': [0, 1],                    # Early low-resolution features
            'second': [2, 3, 4],               # Early-mid resolution features  
            'third': [5, 6, 7],                # Mid-level features
            'fine': [8, 9, 10],                # Fine-grained features
            'low': [11, 12, 13],               # Low-level semantic features
            'mid': [14, 15, 16],               # Mid-level semantic features
            'high': [17]                       # High-level semantic features (256 resolution)
        }
    else:  # resolution == 512
        default_config = {
            'first': [0, 1],                    # Early low-resolution features
            'second': [2, 3, 4],               # Early-mid resolution features  
            'third': [5, 6, 7],                # Mid-level features
            'fine': [8, 9, 10],                # Fine-grained features
            'low': [11, 12, 13],               # Low-level semantic features
            'mid': [14, 15, 16],               # Mid-level semantic features
            'high': [17, 18, 19, 20]           # High-level semantic features (512 resolution)
        }
    
    # Automatically generate simplified overrides based on required indices
    simplified_overrides = {}
    for level_name, default_indices in default_config.items():
        # Find intersection of default indices with required indices
        used_indices = [idx for idx in default_indices if idx in required_indices]
        
        # If we're not using all default indices for this level, create an override
        if used_indices and len(used_indices) < len(default_indices):
            simplified_overrides[level_name] = used_indices
        # If we're not using any indices for this level, keep the default
        # This maintains semantic integrity for unused levels
    
    # Optional: Print configuration for debugging (commented out for production)
    # print(f"Required indices: {required_indices}")
    # print(f"Simplified overrides: {simplified_overrides}")
    
    # Merge configurations: use simplified overrides where specified, default otherwise
    feature_config = default_config.copy()
    feature_config.update(simplified_overrides)
    
    # Validate feature indices
    max_feature_idx = len(features) - 1
    all_indices = []
    for level_indices in feature_config.values():
        all_indices.extend(level_indices)
    
    if max(all_indices) > max_feature_idx:
        raise ValueError(f"Feature index {max(all_indices)} exceeds available features (0-{max_feature_idx})")
    
    # Build feature dictionary by concatenating features at each hierarchical level
    features_dict = {}
    for level_name, indices in feature_config.items():
        if len(indices) == 1:
            # Single feature - no concatenation needed
            features_dict[level_name] = features[indices[0]]
        else:
            # Multiple features - concatenate along channel dimension
            level_features = [features[idx] for idx in indices]
            features_dict[level_name] = torch.cat(level_features, dim=1)

    return features_dict


def collect_features(args, activations_dict: Dict[int, List[torch.Tensor]], sample_idx=0):
    """
    Collect and organize features from different timesteps.
    
    Args:
        args: Configuration arguments containing 'blocks' and 'image_size' fields
        activations_dict: Dictionary mapping timesteps to activation lists
        sample_idx: Sample index to extract from batch
        
    Returns:
        Dictionary mapping timesteps to their organized feature dictionaries
    """
    features_by_t = {}
    
    # Get required indices from configuration blocks
    required_indices = args['blocks']
    
    # Get resolution from args
    resolution = args.get('image_size', 512)
    
    for t, activations in activations_dict.items():
        assert all(isinstance(acts, torch.Tensor) for acts in activations)
        
        # Extract single sample and move to GPU
        resized_activations = [acts[sample_idx][None].to(torch.device('cuda')) for acts in activations]
        
        # Organize features using configuration from blocks
        level_features = prepare_features(resized_activations, required_indices, resolution)
        features_by_t[t] = level_features
        
    return features_by_t