import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset
from torchvision import transforms


class SquatDataset(Dataset):
    """
    Two-stream dataset:
      - original frames (RGB)
      - mediapipe annotated frames (RGB + pose landmarks drawn)
    Folder structure (root_dir):
      root_dir/
        correct/
        fast/
        uncomplete/
        wrong_position/
    """

    def __init__(self, root_dir, hyperparams, split='train'):
        """
        Args:
            root_dir: Path to data directory
            hyperparams: SquatHyperparameters instance
            split: 'train' or 'val' (affects augmentation)
        """
        self.root_dir = str(root_dir)
        self.hp = hyperparams
        self.split = split

        self.classes = self.hp.class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for name in os.listdir(cls_dir):
                if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.video_paths.append(os.path.join(cls_dir, name))
                    self.labels.append(self.class_to_idx[cls])

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.hp.mp_model_complexity,
            min_detection_confidence=self.hp.mp_min_detection_confidence,
            min_tracking_confidence=self.hp.mp_min_tracking_confidence
        )

        # Transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup different transforms for train/val"""
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.hp.img_size, self.hp.img_size)),
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
        
        # Add normalization
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return []

        idxs = np.linspace(0, total - 1, self.hp.num_frames).astype(int)
        frames = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def apply_mediapipe(self, frame_rgb):
        """Apply MediaPipe pose detection and draw landmarks"""
        res = self.pose.process(frame_rgb)
        out = frame_rgb.copy()
        if res.pose_landmarks:
            self.mp_draw.draw_landmarks(
                out,
                res.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return out

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path)

        # FIXED: Instead of returning zeros, try next sample
        if len(frames) == 0:
            print(f"⚠️  Warning: Empty video at {video_path}, trying next sample")
            return self.__getitem__((idx + 1) % len(self))

        original_stream = []
        mediapipe_stream = []

        for fr in frames:
            original_stream.append(self.transform(fr))
            mp_fr = self.apply_mediapipe(fr)
            mediapipe_stream.append(self.transform(mp_fr))

        return {
            "original": torch.stack(original_stream),   # (T, C, H, W)
            "mediapipe": torch.stack(mediapipe_stream), # (T, C, H, W)
            "label": torch.tensor(label, dtype=torch.long)
        }

    def __del__(self):
        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, "pose") and self.pose is not None:
                self.pose.close()
        except Exception:
            pass