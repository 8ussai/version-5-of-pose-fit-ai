import sys
from pathlib import Path

# allow running directly
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import modules.config as config
from modules.training.squats.dataset import SquatDataset
from modules.training.squats.model import TwoStreamSquatClassifier


class SquatTrainer:
    def __init__(self,
                 data_dir=config.DATA_DIR,
                 num_frames=40,
                 img_size=config.IMG_SIZE,
                 batch_size=4,
                 num_epochs=65,
                 learning_rate=0.001,
                 device=None):

        self.data_dir = str(data_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print("Using CUDA device:", torch.cuda.get_device_name(0))

        # outputs
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

        # dataset
        full_dataset = SquatDataset(
            root_dir=self.data_dir,
            num_frames=num_frames,
            img_size=img_size
        )

        # deterministic split (نفس المنطق 80/20 بس مع seed ثابت)
        g = torch.Generator().manual_seed(config.SEED)
        train_size = int(config.TRAIN_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=g
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        self.model = TwoStreamSquatClassifier(num_classes=4).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        self.best_val_acc = 0.0
        self.classes = ["correct", "fast", "uncomplete", "wrong_position"]  

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
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
                'acc': f'{100 * correct / max(1,total):.2f}%'
            })

        epoch_loss = running_loss / max(1, len(self.train_loader))
        epoch_acc = 100 * correct / max(1, total)
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
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
                    'acc': f'{100 * correct / max(1,total):.2f}%'
                })

        epoch_loss = running_loss / max(1, len(self.val_loader))
        epoch_acc = 100 * correct / max(1, total)
        return epoch_loss, epoch_acc, all_preds, all_labels

    def plot_training_history(self):
        # save next to model (outputs/models)
        loss_path = config.MODELS_DIR / f"squats_training_loss.png"
        acc_path  = config.MODELS_DIR / f"squats_training_acc.png"

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Squats - Training/Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        print(f"✓ Training loss plot saved -> {loss_path}")

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Squats - Training/Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(acc_path)
        plt.close()
        print(f"✓ Training acc plot saved -> {acc_path}")

    def train(self):
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)

            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            val_loss, val_acc, _, _ = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.classes,
                    'num_frames': self.num_frames,
                    'img_size': self.img_size,
                    'exercise': "squats",
                    'arch': "twostream_resnet18_bilstm",
                }, config.SQUATS_MODEL)
                print(f"✓ Best model saved -> {config.SQUATS_MODEL} (Val Acc: {val_acc:.2f}%)")

        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*50 + "\n")

        self.plot_training_history()


if __name__ == '__main__':
    trainer = SquatTrainer(
        data_dir=config.DATA_DIR,
        num_frames=40,
        img_size=config.IMG_SIZE,
        batch_size=4,
        num_epochs=65,
        learning_rate=0.001,
        device=None
    )
    trainer.train()
