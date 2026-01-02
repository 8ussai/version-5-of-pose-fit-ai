import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pathlib import Path

import modules.config as config 
from modules.training.shoulder_press.dataset import ShoulderPressNPZDataset
from modules.training.shoulder_press.model import TwoStreamMobileNetGRU


@torch.no_grad()
def main():
    config.ensure_dirs()
    hp = config.SHOULDER_PRESS_HP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shoulder Press paths
    models_dir = config.MODELS_DIR / "shoulder_press"
    eval_dir = config.EVAL_DIR / "shoulder_press"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = config.SHOULDER_PRESS_PROCESSED_DIR

    ckpt_path = models_dir / "shoulder_press_light_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = TwoStreamMobileNetGRU(
        num_classes=len(hp.class_names),
        hidden=hp.gru_hidden,
        layers=hp.gru_layers,
        dropout=hp.dropout,
        freeze_backbone=False  # evaluation ما بتفرق
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ds = ShoulderPressNPZDataset(processed_dir, split="val")
    loader = DataLoader(ds, batch_size=hp.batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = [], []

    for batch in loader:
        x1 = batch["original"].to(device)
        x2 = batch["mediapipe"].to(device)
        y = batch["label"].cpu().numpy()

        logits = model(x1, x2)
        pred = logits.argmax(dim=1).cpu().numpy()

        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(hp.class_names), digits=4)

    print("\n=== Shoulder Press Classification Report ===")
    print(report)

    # plot confusion matrix
    out_png = eval_dir / "confusion_matrix.png"
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title("Shoulder Press - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(hp.class_names)), hp.class_names, rotation=45, ha="right")
    plt.yticks(range(len(hp.class_names)), hp.class_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print(f"\n🖼️ Saved: {out_png}")


if __name__ == "__main__":
    main()