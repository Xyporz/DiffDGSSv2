import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from guided_diffusion_yandex.guided_diffusion.image_datasets import _list_image_files_recursively

import os

def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            transforms.Resize((resolution,resolution)),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform

from typing import List
class FeatureDataset(Dataset):
    """
    Dataset of pixel representations and their labels.
    This dataset loads features for all timesteps for each sample.

    Args:
        y_data: pixel labels [num_pixels]
        class_name: name of the class for feature directory
        timesteps: list of timesteps to load features for
        required_levels: optional set of required feature levels (auto-detected if None)
    """
    def __init__(
        self, 
        y_data: torch.Tensor,
        class_name: str,
        timesteps: List[int],
        required_levels: set = None
    ):    
        self.y_data = y_data
        self.feature_dir = "feature_dir/" + class_name
        self.timesteps = timesteps
        
        # Auto-detect required levels if not provided
        if required_levels is None:
            # Scan the first sample to see what features are available
            self.levels = []
            all_possible_levels = ['first', 'second', 'third', 'fine', 'low', 'mid', 'high']
            for level in all_possible_levels:
                feature_path = os.path.join(self.feature_dir, f"X_{timesteps[0]}_{level}_0.npy")
                if os.path.exists(feature_path):
                    self.levels.append(level)
            print(f"Auto-detected available feature levels: {self.levels}")
        else:
            self.levels = list(required_levels)
            print(f"Using specified feature levels: {self.levels}")
    
    def __getitem__(self, index):
        """
        Returns features for all timesteps for a single sample.
        
        Returns:
            timestep_features: Dictionary containing features for all timesteps, 
                             format: {timestep: {feature_level: feature_tensor}}
            label: Sample label
        """
        # Load features for all timesteps for the current sample
        timestep_features = {}
        
        for t in self.timesteps:
            # Create feature dictionary for each timestep
            features_t = {}
            
            # Load all feature levels
            for level in self.levels:
                feature_path = os.path.join(self.feature_dir, f"X_{t}_{level}_{index}.npy")
                
                try:
                    feature = np.load(feature_path)
                    
                    # Ensure feature is a 3D tensor (C, H, W)
                    if feature.ndim == 4:  # Assume feature saved as (1, C, H, W)
                        feature = feature.squeeze(0)  # Remove batch dimension
                    
                    features_t[level] = torch.from_numpy(feature).float()
                except Exception as e:
                    print(f"Warning: Cannot load feature {feature_path}, error: {e}")
                    # If cannot load feature, return empty tensor and log warning
                    features_t[level] = torch.zeros((1, 1, 1))
            
            # Add current timestep features to timestep feature dictionary
            timestep_features[t] = features_t
        
        return timestep_features, self.y_data[index]

    def __len__(self):
        return len(self.y_data)  # Length equals number of samples, not samples × timesteps


class ImageLabelDataset(Dataset):
    """
    Dataset for images and their corresponding labels.
    
    Args:
        data_dir: Path to folder with images and annotations in *.npy format
        resolution: Image and mask output resolution
        num_images: Restrict number of images in dataset (default: -1 for all)
        transform: Image transforms
    """
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")

        # Load corresponding mask and resize to target resolution
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        tensor_image = self.transform(pil_image)
        tensor_label = torch.from_numpy(label)

        return tensor_image, tensor_label