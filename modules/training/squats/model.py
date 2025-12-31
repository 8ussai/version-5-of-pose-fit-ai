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

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        self.spatial_stream = self._create_cnn_branch()
        self.pose_stream = self._create_cnn_branch()

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
        # Prefer new torchvision API; fallback for older versions
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=True)

        # remove fc layer; output becomes (B, 512, 1, 1)
        layers = list(resnet.children())[:-1]
        return nn.Sequential(*layers)

    def forward(self, original_frames, mediapipe_frames):
        """
        original_frames:  (B, T, C, H, W)
        mediapipe_frames: (B, T, C, H, W)
        returns logits:   (B, num_classes)
        """
        b, t, c, h, w = original_frames.shape

        # spatial stream
        x = original_frames.view(b * t, c, h, w)
        spatial_feat = self.spatial_stream(x).squeeze(-1).squeeze(-1)  # (B*T, 512)
        spatial_feat = spatial_feat.view(b, t, 512)                    # (B, T, 512)

        # pose stream
        y = mediapipe_frames.view(b * t, c, h, w)
        pose_feat = self.pose_stream(y).squeeze(-1).squeeze(-1)        # (B*T, 512)
        pose_feat = pose_feat.view(b, t, 512)                          # (B, T, 512)

        # temporal
        spatial_out, _ = self.temporal_spatial(spatial_feat)  # (B, T, 512) because BiLSTM 256*2
        pose_out, _ = self.temporal_pose(pose_feat)

        spatial_last = spatial_out[:, -1, :]  # (B, 512)
        pose_last = pose_out[:, -1, :]        # (B, 512)

        fused = torch.cat([spatial_last, pose_last], dim=1)  # (B, 1024)
        logits = self.fusion(fused)                          # (B, num_classes)
        return logits
