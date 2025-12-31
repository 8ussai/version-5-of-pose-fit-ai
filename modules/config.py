from pathlib import Path

# Root of repo (مكان هذا الملف)
ROOT_DIR = Path(__file__).resolve().parent

# -------- Paths --------
DATA_DIR = ROOT_DIR / "data" / "squats"

OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EVAL_DIR = OUTPUTS_DIR / "eval"

SQUATS_MODEL = MODELS_DIR / "squats_towstream_resnet18_bilstm.pth"


IMG_SIZE = 224

TRAIN_SPLIT = 0.8
SEED = 42
