import torch
import torch.nn as nn
import torchvision.models as models


class TwoStreamMobileNetGRU(nn.Module):
    """
    Lightweight two-stream:
      - original frames -> MobileNetV2 features (1280)
      - mediapipe frames -> MobileNetV2 features (1280)
      - concat -> GRU
      - classifier
    """

    def __init__(self, num_classes=4, hidden=256, layers=1, dropout=0.3, freeze_backbone=True):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT
        m1 = models.mobilenet_v2(weights=weights)
        m2 = models.mobilenet_v2(weights=weights)

        self.spatial = m1.features
        self.pose = m2.features

        self.gap = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280

        if freeze_backbone:
            for p in self.spatial.parameters():
                p.requires_grad = False
            for p in self.pose.parameters():
                p.requires_grad = False

        self.temporal = nn.GRU(
            input_size=feat_dim * 2,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, original_frames, mediapipe_frames):
        # original_frames: (B,T,C,H,W)
        B, T, C, H, W = original_frames.shape

        x1 = original_frames.view(B * T, C, H, W)
        x2 = mediapipe_frames.view(B * T, C, H, W)

        f1 = self.spatial(x1)                  # (B*T,1280,7,7)
        f1 = self.gap(f1).flatten(1)           # (B*T,1280)

        f2 = self.pose(x2)
        f2 = self.gap(f2).flatten(1)           # (B*T,1280)

        fused = torch.cat([f1, f2], dim=1)     # (B*T,2560)
        fused = fused.view(B, T, -1)           # (B,T,2560)

        out, _ = self.temporal(fused)          # (B,T,H)
        last = out[:, -1, :]                   # (B,H)

        return self.classifier(last)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
