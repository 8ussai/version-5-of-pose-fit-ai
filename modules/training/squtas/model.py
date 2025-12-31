import torch
import torch.nn as nn
import torchvision.models as models

class TwoStreamSquatClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        """
        Two-Stream Architecture للسكوات
        
        Args:
            num_classes: عدد الفئات (4 في حالتنا)
            dropout: نسبة الـ dropout
        """
        super(TwoStreamSquatClassifier, self).__init__()
        
        # Stream 1: Original Video Branch (Spatial Stream)
        self.spatial_stream = self._create_cnn_branch()
        
        # Stream 2: MediaPipe Visual Branch (Pose Stream)
        self.pose_stream = self._create_cnn_branch()
        
        # Temporal processing (LSTM/GRU)
        self.temporal_spatial = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.temporal_pose = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Fusion layer
        # 256*2 (bidirectional) * 2 (two streams) = 1024
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _create_cnn_branch(self):
        """إنشاء CNN branch باستخدام ResNet18 pretrained"""
        # استخدام ResNet18 كـ feature extractor
        resnet = models.resnet18(pretrained=True)
        
        # إزالة الـ fully connected layer الأخيرة
        modules = list(resnet.children())[:-1]
        cnn = nn.Sequential(*modules)
        
        return cnn
    
    def forward(self, original_frames, mediapipe_frames):
        """
        Args:
            original_frames: [batch, num_frames, channels, height, width]
            mediapipe_frames: [batch, num_frames, channels, height, width]
        
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, num_frames, c, h, w = original_frames.shape
        
        # معالجة Original Stream
        # تحويل لـ [batch * num_frames, c, h, w] لمعالجة كل frame
        original_frames = original_frames.view(batch_size * num_frames, c, h, w)
        spatial_features = self.spatial_stream(original_frames)  # [batch*num_frames, 512, 1, 1]
        spatial_features = spatial_features.squeeze(-1).squeeze(-1)  # [batch*num_frames, 512]
        spatial_features = spatial_features.view(batch_size, num_frames, -1)  # [batch, num_frames, 512]
        
        # معالجة MediaPipe Stream
        mediapipe_frames = mediapipe_frames.view(batch_size * num_frames, c, h, w)
        pose_features = self.pose_stream(mediapipe_frames)  # [batch*num_frames, 512, 1, 1]
        pose_features = pose_features.squeeze(-1).squeeze(-1)  # [batch*num_frames, 512]
        pose_features = pose_features.view(batch_size, num_frames, -1)  # [batch, num_frames, 512]
        
        # Temporal processing
        spatial_temporal, _ = self.temporal_spatial(spatial_features)  # [batch, num_frames, 512]
        pose_temporal, _ = self.temporal_pose(pose_features)  # [batch, num_frames, 512]
        
        # أخذ آخر output من LSTM
        spatial_final = spatial_temporal[:, -1, :]  # [batch, 512]
        pose_final = pose_temporal[:, -1, :]  # [batch, 512]
        
        # Fusion
        fused = torch.cat([spatial_final, pose_final], dim=1)  # [batch, 1024]
        
        # Classification
        logits = self.fusion(fused)  # [batch, num_classes]
        
        return logits