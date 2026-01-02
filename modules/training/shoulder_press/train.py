import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

import modules.config as config
from modules.training.shoulder_press.dataset import ShoulderPressNPZDataset
from modules.training.shoulder_press.model import TwoStreamMobileNetGRU


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self):
        config.ensure_dirs()
        self.hp = config.SHOULDER_PRESS_HP
        set_seed(config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Device: {self.device}")
        if self.device.type == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

        # Shoulder Press paths
        self.processed_dir = config.SHOULDER_PRESS_PROCESSED_DIR
        self.models_dir = config.MODELS_DIR / "shoulder_press"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Dataset (NPZ)
        if not self.processed_dir.exists() or len(list(self.processed_dir.glob("*.npz"))) == 0:
            raise FileNotFoundError(
                f"❌ No npz found in {self.processed_dir}\n"
                f"Run first: python modules/training/shoulder_press/preprocess_videos.py"
            )

        full = ShoulderPressNPZDataset(self.processed_dir, split="train")

        g = torch.Generator().manual_seed(config.SEED)
        train_size = int(config.TRAIN_SPLIT * len(full))
        val_size = len(full) - train_size
        train_ds, val_ds = random_split(full, [train_size, val_size], generator=g)

        # validation dataset: نفس الملفات بس بدون augmentation
        full_val = ShoulderPressNPZDataset(self.processed_dir, split="val")
        _, val_ds = random_split(full_val, [train_size, val_size], generator=g)

        self.train_loader = DataLoader(
            train_ds, batch_size=self.hp.batch_size, shuffle=True,
            num_workers=self.hp.num_workers, pin_memory=(self.device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.hp.batch_size, shuffle=False,
            num_workers=self.hp.num_workers, pin_memory=(self.device.type == "cuda")
        )

        # Model (Lightweight only)
        self.model = TwoStreamMobileNetGRU(
            num_classes=len(self.hp.class_names),
            hidden=self.hp.gru_hidden,
            layers=self.hp.gru_layers,
            dropout=self.hp.dropout,
            freeze_backbone=self.hp.freeze_backbone
        ).to(self.device)

        print(f"🏗️ Params: {self.model.num_params():,}")
        print(f"✅ Trainable: {self.model.num_trainable_params():,}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        # AMP (new API)
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.best_val_acc = 0.0
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def _accuracy(self, logits, y):
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            x1 = batch["original"].to(self.device)
            x2 = batch["mediapipe"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x1, x2)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            acc = self._accuracy(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n += bs

            pbar.set_postfix(loss=total_loss / n, acc=total_acc / n)

        return total_loss / n, total_acc / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            x1 = batch["original"].to(self.device)
            x2 = batch["mediapipe"].to(self.device)
            y = batch["label"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x1, x2)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)

            acc = self._accuracy(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n += bs

        return total_loss / n, total_acc / n

    def plot_curves(self):
        out_dir = config.OUTPUTS_DIR / "shoulder_press"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_png = out_dir / "training_curves.png"
        plt.figure()
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.val_losses, label="val_loss")
        plt.legend()
        plt.title("Shoulder Press - Loss Curves")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

        out_png2 = out_dir / "accuracy_curves.png"
        plt.figure()
        plt.plot(self.train_accs, label="train_acc")
        plt.plot(self.val_accs, label="val_acc")
        plt.legend()
        plt.title("Shoulder Press - Accuracy Curves")
        plt.savefig(out_png2, bbox_inches="tight")
        plt.close()

        print(f"📈 Saved: {out_png}")
        print(f"📈 Saved: {out_png2}")

    def fit(self):
        best_path = self.models_dir / "shoulder_press_light_best.pt"

        for epoch in range(1, self.hp.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.hp.num_epochs}")
            tr_loss, tr_acc = self.train_one_epoch()
            va_loss, va_acc = self.validate()

            self.scheduler.step(va_loss)

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.train_accs.append(tr_acc)
            self.val_accs.append(va_acc)

            print(f"✅ train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}")
            print(f"✅ val_loss  ={va_loss:.4f} val_acc  ={va_acc:.4f}")

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "hp": self.hp.__dict__,
                        "class_names": self.hp.class_names,
                    },
                    best_path
                )
                print(f"💾 Saved best -> {best_path} (val_acc={va_acc:.4f})")

        self.plot_curves()
        print(f"\n🏁 Best val acc: {self.best_val_acc:.4f}")


if __name__ == "__main__":
    Trainer().fit()