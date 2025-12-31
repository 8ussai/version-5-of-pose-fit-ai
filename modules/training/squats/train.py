import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import modules.config as config
from modules.training.squats.dataset import SquatDataset
from modules.training.squats.model import TwoStreamSquatClassifier


class SquatTrainer:
    def __init__(self, hyperparams=None, device=None):
        """
        Args:
            hyperparams: SquatHyperparameters instance (uses default if None)
            device: torch device (auto-detects if None)
        """
        self.hp = hyperparams if hyperparams is not None else config.SQUAT_HP
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"🖥️  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

        # Create output directories
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

        # Print hyperparameters
        self.hp.print_summary()

        # Setup datasets
        self._setup_datasets()
        
        # Setup model
        self.model = TwoStreamSquatClassifier(self.hp).to(self.device)
        print(f"\n🏗️  Model Parameters: {self.model.get_num_params():,}")

        # Setup training components
        self._setup_training()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0

    def _setup_datasets(self):
        """Setup train and validation datasets"""
        full_dataset = SquatDataset(
            root_dir=config.DATA_DIR,
            hyperparams=self.hp,
            split='train'
        )

        # Deterministic split
        g = torch.Generator().manual_seed(config.SEED)
        train_size = int(config.TRAIN_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size], generator=g
        )

        # Update val dataset to use val transforms (no augmentation)
        val_dataset_proper = SquatDataset(
            root_dir=config.DATA_DIR,
            hyperparams=self.hp,
            split='val'
        )
        # Copy the indices from val_ds
        val_dataset_proper.video_paths = [full_dataset.video_paths[i] for i in val_ds.indices]
        val_dataset_proper.labels = [full_dataset.labels[i] for i in val_ds.indices]

        print(f"\n📊 Dataset Split:")
        print(f"   Train: {len(train_ds)} samples")
        print(f"   Val:   {len(val_dataset_proper)} samples")

        # Setup data loaders
        num_workers = 2 if self.device.type == "cuda" else 0
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )

        self.val_loader = DataLoader(
            val_dataset_proper,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )

    def _setup_training(self):
        """Setup loss, optimizer, and scheduler"""
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer selection
        if self.hp.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.hp.learning_rate,
                weight_decay=self.hp.weight_decay
            )
        elif self.hp.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.hp.learning_rate,
                weight_decay=self.hp.weight_decay
            )
        elif self.hp.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.hp.learning_rate,
                momentum=0.9,
                weight_decay=self.hp.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hp.optimizer_type}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.hp.scheduler_patience,
            factor=self.hp.scheduler_factor,
            min_lr=self.hp.scheduler_min_lr
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', ncols=100)
        for batch in pbar:
            original = batch['original'].to(self.device)
            mediapipe = batch['mediapipe'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(original, mediapipe)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / max(1, total):.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc='Validation', ncols=100)
        for batch in pbar:
            original = batch['original'].to(self.device)
            mediapipe = batch['mediapipe'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(original, mediapipe)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / max(1, total):.2f}%'
            })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels

    def plot_training_history(self):
        """Plot and save training curves"""
        loss_path = config.MODELS_DIR / "squats_training_loss.png"
        acc_path = config.MODELS_DIR / "squats_training_acc.png"

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Squats - Training/Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"✅ Loss plot saved: {loss_path}")

        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accs, label='Train Acc', linewidth=2)
        plt.plot(self.val_accs, label='Val Acc', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Squats - Training/Validation Accuracy', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(acc_path, dpi=150)
        plt.close()
        print(f"✅ Accuracy plot saved: {acc_path}")

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("🚀 STARTING TRAINING")
        print("=" * 70 + "\n")

        for epoch in range(self.hp.num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.hp.num_epochs}")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, _, _ = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print summary
            print(f"\n📊 Epoch Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'hyperparameters': self.hp,
                    'class_names': self.hp.class_names,
                }
                torch.save(checkpoint, config.SQUATS_MODEL)
                print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")

        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED!")
        print(f"🏆 Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("=" * 70 + "\n")

        self.plot_training_history()


def main():
    """Main entry point"""
    # You can customize hyperparameters here
    hp = config.SQUAT_HP  # Use default
    
    # Or create custom hyperparameters:
    # from modules.config import SquatHyperparameters
    # hp = SquatHyperparameters(
    #     batch_size=8,
    #     num_epochs=50,
    #     learning_rate=0.0005,
    # )
    
    trainer = SquatTrainer(hyperparams=hp)
    trainer.train()


if __name__ == '__main__':
    main()