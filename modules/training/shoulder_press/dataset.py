import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class ShoulderPressNPZDataset(Dataset):
    """
    Loads preprocessed .npz files for shoulder press:
      - original:  (T,H,W,C)
      - mediapipe: (T,H,W,C)
      - label: int
    """

    def __init__(self, processed_dir, split="train"):
        self.processed_dir = Path(processed_dir)
        import modules.config as config
        self.hp = config.SHOULDER_PRESS_HP
        self.split = split

        self.npz_files = sorted(list(self.processed_dir.glob("*.npz")))
        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in: {self.processed_dir}")

        self._setup_transforms()

    def _setup_transforms(self):
        t = [transforms.ToPILImage()]

        if self.split == "train" and self.hp.use_augmentation:
            t += [
                transforms.RandomHorizontalFlip(p=self.hp.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=self.hp.color_jitter_brightness,
                    contrast=self.hp.color_jitter_contrast
                )
            ]

        t += [
            transforms.Resize((self.hp.img_size, self.hp.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        data = np.load(npz_path)

        original_frames = data["original"]      # (T,H,W,C)
        mediapipe_frames = data["mediapipe"]    # (T,H,W,C)
        label = int(data["label"])

        orig_stack, mp_stack = [], []
        for i in range(original_frames.shape[0]):
            orig_stack.append(self.transform(original_frames[i]))
            mp_stack.append(self.transform(mediapipe_frames[i]))

        return {
            "original": torch.stack(orig_stack),    # (T,C,H,W)
            "mediapipe": torch.stack(mp_stack),     # (T,C,H,W)
            "label": torch.tensor(label, dtype=torch.long)
        }