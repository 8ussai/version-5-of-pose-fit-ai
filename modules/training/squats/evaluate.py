import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

import modules.config as config
from modules.training.squats.dataset import SquatDataset
from modules.training.squats.model import TwoStreamSquatClassifier

CLASS_NAMES = ["correct", "fast", "uncomplete", "wrong_position"]

@torch.no_grad()
def main():
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset + deterministic split (نفس train.py)
    ds = SquatDataset(
        root_dir=str(config.DATA_DIR),
        num_frames=40,
        img_size=config.IMG_SIZE
    )

    g = torch.Generator().manual_seed(config.SEED)
    train_size = int(config.TRAIN_SPLIT * len(ds))
    val_size = len(ds) - train_size
    _, val_ds = random_split(ds, [train_size, val_size], generator=g)

    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    if not config.SQUATS_MODEL.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.SQUATS_MODEL}")

    model = TwoStreamSquatClassifier(num_classes=4).to(device)
    ckpt = torch.load(config.SQUATS_MODEL, map_location=device)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    for batch in loader:
        original = batch['original'].to(device)
        mediapipe = batch['mediapipe'].to(device)
        labels = batch['label'].to(device)

        logits = model(original, mediapipe)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3)
    print("\nClassification Report:\n", report)

    report_path = config.EVAL_DIR / "squats_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("✓ Saved:", report_path)

    # confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('squats - Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = config.EVAL_DIR / "squats_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print("✓ Saved:", cm_path)


if __name__ == "__main__":
    main()
