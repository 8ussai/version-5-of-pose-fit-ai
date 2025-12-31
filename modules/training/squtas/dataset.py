import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset
from torchvision import transforms

class SquatDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, img_size=224, transform=None):
        """
        Dataset للسكوات مع معالجة MediaPipe on-the-fly
        
        Args:
            root_dir: مسار مجلد data/squats
            num_frames: عدد الإطارات المستخرجة من كل فيديو
            img_size: حجم الصورة للـ model
            transform: تحويلات إضافية
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform
        
        # تحضير قائمة الفيديوهات والتصنيفات
        self.classes = ['correct', 'fast', 'uncomplete', 'wrong_position']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.video_paths = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir):
                for video_name in os.listdir(cls_dir):
                    if video_name.endswith(('.mp4', '.avi', '.mov')):
                        self.video_paths.append(os.path.join(cls_dir, video_name))
                        self.labels.append(self.class_to_idx[cls])
        
        # تهيئة MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Transform افتراضي
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """استخراج frames من الفيديو بشكل متساوي"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def apply_mediapipe(self, frame):
        """تطبيق MediaPipe ورسم الـ skeleton على الـ frame"""
        # معالجة الصورة مع MediaPipe
        results = self.pose.process(frame)
        
        # نسخ الصورة للرسم عليها
        annotated_frame = frame.copy()
        
        # رسم الـ pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
            )
        
        return annotated_frame
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # استخراج الـ frames
        frames = self.extract_frames(video_path)
        
        # Stream 1: Original frames
        original_stream = []
        # Stream 2: MediaPipe overlayed frames
        mediapipe_stream = []
        
        for frame in frames:
            # Original frame
            original_tensor = self.transform(frame)
            original_stream.append(original_tensor)
            
            # MediaPipe frame
            mp_frame = self.apply_mediapipe(frame)
            mp_tensor = self.transform(mp_frame)
            mediapipe_stream.append(mp_tensor)
        
        # تحويل لـ tensors: [num_frames, channels, height, width]
        original_stream = torch.stack(original_stream)
        mediapipe_stream = torch.stack(mediapipe_stream)
        
        return {
            'original': original_stream,
            'mediapipe': mediapipe_stream,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def __del__(self):
        """تنظيف MediaPipe عند انتهاء الاستخدام"""
        if hasattr(self, 'pose'):
            self.pose.close()