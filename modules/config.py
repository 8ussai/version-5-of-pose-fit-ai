from dataclasses import dataclass
from pathlib import Path

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Squats paths
DATA_DIR = PROJECT_ROOT / "data" / "squats"
PROCESSED_DIR = PROJECT_ROOT / "data" / "squats_processed"

# Shoulder Press paths
SHOULDER_PRESS_DATA_DIR = PROJECT_ROOT / "data" / "shoulder_press"
SHOULDER_PRESS_PROCESSED_DIR = PROJECT_ROOT / "data" / "shoulder_press_processed"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EVAL_DIR = OUTPUTS_DIR / "eval"

SEED = 42
TRAIN_SPLIT = 0.8

# =========================
# Hyperparameters
# =========================
@dataclass
class SquatHyperparameters:
    # data
    class_names: tuple = ("correct", "fast", "incomplete", "wrong_position")
    num_frames: int = 40
    img_size: int = 224

    # training
    batch_size: int = 4
    num_epochs: int = 45
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0  # Windows: خليها 0

    # augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

    # model (lightweight)
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.3
    freeze_backbone: bool = True  # لايت ويت أكثر

    # realtime params (اختياري)
    webcam_id: int = 0
    mp_model_complexity: int = 1
    calib_seconds: float = 2.0
    up_offset: float = 0.10
    down_offset: float = 0.10
    smoothing: int = 5


@dataclass
class ShoulderPressHyperparameters:
    # data
    class_names: tuple = ("correct", "fast", "incomplete", "wrong_position")
    num_frames: int = 41
    img_size: int = 224

    # training
    batch_size: int = 4
    num_epochs: int = 45
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0  # Windows: خليها 0

    # augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

    # model (lightweight)
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.3
    freeze_backbone: bool = True

    # realtime params
    webcam_id: int = 0
    mp_model_complexity: int = 1
    calib_seconds: float = 2.0
    up_offset: float = 0.15  # أعلى شوي للشولدر
    down_offset: float = 0.15
    smoothing: int = 5


# Instances
HP = SquatHyperparameters()
SHOULDER_PRESS_HP = ShoulderPressHyperparameters()


def ensure_dirs():
    """Create all necessary directories"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SHOULDER_PRESS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for models and eval
    (MODELS_DIR / "squats").mkdir(exist_ok=True)
    (MODELS_DIR / "shoulder_press").mkdir(exist_ok=True)
    (EVAL_DIR / "squats").mkdir(exist_ok=True)
    (EVAL_DIR / "shoulder_press").mkdir(exist_ok=True)