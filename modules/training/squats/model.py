import torch
import torch.nn as nn
import torchvision.models as models


class TwoStreamSquatClassifier(nn.Module):
    """
    Two-stream model:
      - spatial stream: ResNet18 -> 512 features per frame
      - pose stream:    ResNet18 -> 512 features per frame
      - temporal: BiLSTM on each stream
      - fusion: concat last-step features -> MLP classifier
    """

    def __init__(self, hyperparams):
        """
        Args:
            hyperparams: SquatHyperparameters instance
        """
        super().__init__()
        
        self.hp = hyperparams

        # CNN branches
        self.spatial_stream = self._create_cnn_branch()
        self.pose_stream = self._create_cnn_branch()

        # Temporal modules
        self.temporal_spatial = nn.LSTM(
            input_size=512,
            hidden_size=self.hp.lstm_hidden_size,
            num_layers=self.hp.lstm_num_layers,
            batch_first=True,
            dropout=self.hp.dropout if self.hp.lstm_num_layers > 1 else 0,
            bidirectional=self.hp.lstm_bidirectional
        )

        self.temporal_pose = nn.LSTM(
            input_size=512,
            hidden_size=self.hp.lstm_hidden_size,
            num_layers=self.hp.lstm_num_layers,
            batch_first=True,
            dropout=self.hp.dropout if self.hp.lstm_num_layers > 1 else 0,
            bidirectional=self.hp.lstm_bidirectional
        )

        # Fusion layer (simplified to reduce overfitting)
        fusion_input_size = self.hp.get_fusion_input_size()
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(self.hp.dropout),
            nn.Linear(256, self.hp.num_classes)
        )

    def _create_cnn_branch(self):
        """Create ResNet18 feature extractor"""
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=True)

        # Remove fc layer; output becomes (B, 512, 1, 1)
        layers = list(resnet.children())[:-1]
        return nn.Sequential(*layers)

    def forward(self, original_frames, mediapipe_frames):
        """
        Args:
            original_frames:  (B, T, C, H, W)
            mediapipe_frames: (B, T, C, H, W)
        
        Returns:
            logits: (B, num_classes)
        """
        b, t, c, h, w = original_frames.shape

        # Spatial stream
        x = original_frames.view(b * t, c, h, w)
        spatial_feat = self.spatial_stream(x).squeeze(-1).squeeze(-1)  # (B*T, 512)
        spatial_feat = spatial_feat.view(b, t, 512)                    # (B, T, 512)

        # Pose stream
        y = mediapipe_frames.view(b * t, c, h, w)
        pose_feat = self.pose_stream(y).squeeze(-1).squeeze(-1)        # (B*T, 512)
        pose_feat = pose_feat.view(b, t, 512)                          # (B, T, 512)

        # Temporal processing
        spatial_out, _ = self.temporal_spatial(spatial_feat)  # (B, T, hidden*directions)
        pose_out, _ = self.temporal_pose(pose_feat)

        # Take last timestep
        spatial_last = spatial_out[:, -1, :]  # (B, hidden*directions)
        pose_last = pose_out[:, -1, :]        # (B, hidden*directions)

        # Fusion
        fused = torch.cat([spatial_last, pose_last], dim=1)
        logits = self.fusion(fused)
        
        return logits
    
    def get_num_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)