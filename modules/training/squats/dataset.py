"""
Dataset that loads pre-processed videos (avoids MediaPipe during training)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class SquatDatasetPreprocessed(Dataset):
    """
    Dataset that loads pre-processed video frames from .npz files.
    This avoids MediaPipe initialization issues during training.
    """

    def __init__(self, processed_dir, hyperparams, split='train'):
        """
        Args:
            processed_dir: Path to directory with pre-processed .npz files
            hyperparams: SquatHyperparameters instance
            split: 'train' or 'val' (affects augmentation)
        """
        self.processed_dir = Path(processed_dir)
        self.hp = hyperparams
        self.split = split

        # Find all .npz files
        self.npz_files = list(self.processed_dir.glob("*.npz"))
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in {processed_dir}")

        print(f"📂 Found {len(self.npz_files)} pre-processed videos")

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup different transforms for train/val"""
        base_transforms = [
            transforms.ToPILImage(),
        ]
        
        # Add augmentation for training only
        if self.split == 'train' and self.hp.use_augmentation:
            base_transforms.extend([
                transforms.RandomHorizontalFlip(p=self.hp.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=self.hp.color_jitter_brightness,
                    contrast=self.hp.color_jitter_contrast
                ),
            ])
        
        # Resize and normalize
        base_transforms.extend([
            transforms.Resize((self.hp.img_size, self.hp.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        
        try:
            # Load pre-processed data
            data = np.load(npz_path)
            original_frames = data['original']      # (T, H, W, C)
            mediapipe_frames = data['mediapipe']    # (T, H, W, C)
            label = int(data['label'])
            
            # Apply transforms to each frame
            original_stream = []
            mediapipe_stream = []
            
            for i in range(len(original_frames)):
                original_stream.append(self.transform(original_frames[i]))
                mediapipe_stream.append(self.transform(mediapipe_frames[i]))
            
            return {
                "original": torch.stack(original_stream),    # (T, C, H, W)
                "mediapipe": torch.stack(mediapipe_stream),  # (T, C, H, W)
                "label": torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"⚠️  Error loading {npz_path}: {e}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))