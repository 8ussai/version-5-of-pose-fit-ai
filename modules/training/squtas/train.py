import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from modules.training.squtas.dataset import SquatDataset
from modules.training.squtas.model import TwoStreamSquatClassifier

class SquatTrainer:
    def __init__(self, 
                 data_dir='data/squats',
                 num_frames=16,
                 img_size=224,
                 batch_size=4,
                 num_epochs=65,
                 learning_rate=0.001,
                 device='cuda'):
        
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0")
        print("Using CUDA device:", torch.cuda.get_device_name(0))

        
        print(f"Using device: {self.device}")
        
        # تحضير الـ dataset
        full_dataset = SquatDataset(
            root_dir=data_dir,
            num_frames=num_frames,
            img_size=img_size
        )
        
        # تقسيم لـ train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.model = TwoStreamSquatClassifier(num_classes=4).to(self.device)
        
        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        self.best_val_acc = 0.0
        self.classes = ['correct', 'fast', 'uncomplete', 'wrong_position']
    
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
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(original, mediapipe)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
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
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self):
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50 + "\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pth')
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        
        print("\n" + "="*50)
        print("Training Completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*50 + "\n")
        
        # Final evaluation
        self.plot_training_history()
        self.evaluate_model()
    
    def plot_training_history(self):
        """رسم تطور التدريب"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("✓ Training history saved to 'training_history.png'")
    
    def evaluate_model(self):
        """تقييم نهائي مع confusion matrix"""
        print("\nFinal Evaluation on Validation Set:")
        print("-" * 50)
        
        _, _, val_preds, val_labels = self.validate()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(val_labels, val_preds, 
                                   target_names=self.classes,
                                   digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        
        # إضافة الأرقام على المصفوفة
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")


if __name__ == '__main__':
    # Configuration
    trainer = SquatTrainer(
        data_dir='data/squats',
        num_frames=16,          # عدد الإطارات من كل فيديو
        img_size=224,           # حجم الصورة
        batch_size=4,           # حجم الـ batch (قلله إذا نفذت الذاكرة)
        num_epochs=65,          # عدد الـ epochs
        learning_rate=0.001,
        device='cuda'
    )
    
    # Start training
    trainer.train()