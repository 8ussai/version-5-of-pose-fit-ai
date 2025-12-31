import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import modules.config as config
from modules.training.squats.dataset import SquatDataset
from modules.training.squats.model import TwoStreamSquatClassifier


@torch.no_grad()
def evaluate_model():
    """Evaluate trained model on validation set"""
    
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")

    # Check if model exists
    if not config.SQUATS_MODEL.exists():
        raise FileNotFoundError(f"❌ Model not found: {config.SQUATS_MODEL}")

    # Load checkpoint
    print(f"📂 Loading model from: {config.SQUATS_MODEL}")
    ckpt = torch.load(config.SQUATS_MODEL, map_location=device)
    
    # Extract hyperparameters from checkpoint if available
    if isinstance(ckpt, dict) and 'hyperparameters' in ckpt:
        hp = ckpt['hyperparameters']
        print("✅ Loaded hyperparameters from checkpoint")
    else:
        hp = config.SQUAT_HP
        print("⚠️  Using default hyperparameters (not found in checkpoint)")
    
    hp.print_summary()

    # Create dataset (same split as training)
    full_dataset = SquatDataset(
        root_dir=config.DATA_DIR,
        hyperparams=hp,
        split='val'  # No augmentation for evaluation
    )

    g = torch.Generator().manual_seed(config.SEED)
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=g)

    print(f"\n📊 Validation samples: {len(val_ds)}")

    # Create data loader
    loader = DataLoader(
        val_ds,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create and load model
    model = TwoStreamSquatClassifier(hp).to(device)
    
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"🏗️  Parameters: {model.get_num_params():,}\n")

    # Inference
    print("🔄 Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        original = batch['original'].to(device)
        mediapipe = batch['mediapipe'].to(device)
        labels = batch['label'].to(device)

        logits = model(original, mediapipe)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("✅ Inference complete!\n")
    
    # Overall accuracy
    accuracy = 100 * np.mean(all_preds == all_labels)
    print(f"🎯 Overall Accuracy: {accuracy:.2f}%\n")

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=hp.class_names,
        digits=3
    )
    print("📋 Classification Report:")
    print(report)

    # Save report
    report_path = config.EVAL_DIR / "squats_classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Squat Classifier - Evaluation Report\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write(f"Classification Report:\n")
        f.write(report)
    print(f"✅ Report saved: {report_path}\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=hp.class_names,
        yticklabels=hp.class_names,
        title='Squats - Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold'
            )
    
    fig.tight_layout()
    cm_path = config.EVAL_DIR / "squats_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")

    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for i, cls_name in enumerate(hp.class_names):
        cls_mask = all_labels == i
        if cls_mask.sum() > 0:
            cls_acc = 100 * np.mean(all_preds[cls_mask] == all_labels[cls_mask])
            print(f"   {cls_name:20s}: {cls_acc:.2f}% ({cls_mask.sum()} samples)")

    # Macro F1 score
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\n🎯 Macro F1 Score: {f1:.3f}")

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    evaluate_model()