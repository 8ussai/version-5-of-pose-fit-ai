from pathlib import Path
from dataclasses import dataclass
import torch
import platform


# ============================================
# PROJECT ROOT
# ============================================
# This file is at: project_root/modules/config.py
ROOT_DIR = Path(__file__).resolve().parent.parent

# ============================================
# DIRECTORY STRUCTURE
# ============================================
DATA_DIR = ROOT_DIR / "data" / "squats"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EVAL_DIR = OUTPUTS_DIR / "eval"

# ============================================
# MODEL PATHS
# ============================================
SQUATS_MODEL = MODELS_DIR / "squats_twostream_resnet18_bilstm.pth"

# ============================================
# GLOBAL SETTINGS
# ============================================
IMG_SIZE = 224
TRAIN_SPLIT = 0.8
SEED = 42


# ============================================
# SQUATS HYPERPARAMETERS CLASS
# ============================================
@dataclass
class SquatHyperparameters:
    """All hyperparameters for squat training and inference"""
    
    # -------- Data & Preprocessing --------
    num_frames: int = 40
    img_size: int = 224
    num_classes: int = 4
    class_names: list = None
    
    # -------- Training --------
    batch_size: int = 4
    num_epochs: int = 65
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_workers: int = 0  # Set to 0 for Windows compatibility
    
    # -------- Model Architecture --------
    dropout: float = 0.5
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # -------- Optimizer & Scheduler --------
    optimizer_type: str = "adam"  # adam, sgd, adamw
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # -------- Data Augmentation --------
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    
    # -------- MediaPipe Settings --------
    mp_model_complexity: int = 1
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5
    
    # -------- Real-Time Inference --------
    calib_seconds: float = 5.0
    calib_min_samples: int = 15
    up_offset: float = 0.02
    down_offset: float = 0.08
    result_hold_sec: float = 2.0
    camera_index: int = 0
    
    # -------- Visualization Colors (BGR) --------
    colors: dict = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.class_names is None:
            self.class_names = ["correct", "fast", "incomplete", "wrong_position"]
        
        if self.colors is None:
            self.colors = {
                "correct": (0, 255, 0),           # Green
                "fast": (0, 255, 255),            # Yellow
                "incomplete": (0, 165, 255),      # Orange
                "wrong_position": (0, 0, 255),    # Red
            }
    
    def get_fusion_input_size(self) -> int:
        """Calculate fusion layer input size based on LSTM config"""
        multiplier = 2 if self.lstm_bidirectional else 1
        return self.lstm_hidden_size * multiplier * 2  # *2 for two streams
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("SQUAT CLASSIFIER - HYPERPARAMETERS")
        print("=" * 60)
        print(f"📊 Data:")
        print(f"   - Num Frames: {self.num_frames}")
        print(f"   - Image Size: {self.img_size}")
        print(f"   - Classes: {self.class_names}")
        print(f"\n🎓 Training:")
        print(f"   - Batch Size: {self.batch_size}")
        print(f"   - Epochs: {self.num_epochs}")
        print(f"   - Learning Rate: {self.learning_rate}")
        print(f"   - Optimizer: {self.optimizer_type.upper()}")
        print(f"\n🏗️  Architecture:")
        print(f"   - Dropout: {self.dropout}")
        print(f"   - LSTM Hidden: {self.lstm_hidden_size}")
        print(f"   - LSTM Layers: {self.lstm_num_layers}")
        print(f"   - Bidirectional: {self.lstm_bidirectional}")
        print(f"   - Fusion Input Size: {self.get_fusion_input_size()}")
        print(f"\n🎥 Real-Time:")
        print(f"   - Calibration: {self.calib_seconds}s")
        print(f"   - UP Offset: {self.up_offset}")
        print(f"   - DOWN Offset: {self.down_offset}")
        print("=" * 60)


# ============================================
# DEFAULT INSTANCE
# ============================================
SQUAT_HP = SquatHyperparameters()


# ============================================
# SYSTEM INFO
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU Only"


# ============================================
# PATH VALIDATION
# ============================================
def validate_paths():
    """Validate and create required directories"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_DIR.exists():
        print(f"⚠️  Warning: Data directory not found: {DATA_DIR}")
        print(f"   Create it and add training videos in subdirectories:")
        for cls in SQUAT_HP.class_names:
            print(f"   - {DATA_DIR / cls}/")
    
    return True


# Run validation on import
validate_paths()


# ============================================
# MAIN - Print Config Summary
# ============================================
if __name__ == "__main__":
    print("\n📁 Project Structure:")
    print(f"   Root: {ROOT_DIR}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Eval: {EVAL_DIR}")
    print(f"\n💻 System:")
    print(f"   Device: {DEVICE}")
    print(f"   GPU: {GPU_NAME}")
    print()
    SQUAT_HP.print_summary()