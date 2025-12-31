import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms

import modules.config as config
from modules.training.squats.model import TwoStreamSquatClassifier


WINDOW_NAME = "Squat - One Rep Judge (Real Time)"


def sample_uniform(frames, n=16):
    """Sample n frames uniformly from list"""
    if len(frames) == 0:
        return []
    if len(frames) == n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def safe_load_checkpoint(model, checkpoint_path, device):
    """Safely load model checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        # Return hyperparameters if available
        hp = ckpt.get('hyperparameters', config.SQUAT_HP)
    else:
        model.load_state_dict(ckpt)
        hp = config.SQUAT_HP
    
    model.to(device)
    model.eval()
    return hp


def main():
    # Load hyperparameters from checkpoint or use defaults
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("🎥 SQUAT REAL-TIME INFERENCE")
    print("="*70)
    
    # Create model with default hyperparams first (will be updated after loading)
    model = TwoStreamSquatClassifier(config.SQUAT_HP).to(device)
    
    # Load checkpoint and get actual hyperparameters
    print(f"\n📂 Loading model: {config.SQUATS_MODEL}")
    hp = safe_load_checkpoint(model, config.SQUATS_MODEL, device)
    
    print(f"✅ Model loaded successfully")
    print(f"🖥️  Device: {device}")
    print(f"🏗️  Parameters: {model.get_num_params():,}")
    
    # Print key inference settings
    print(f"\n⚙️  Inference Settings:")
    print(f"   Num Frames: {hp.num_frames}")
    print(f"   Image Size: {hp.img_size}")
    print(f"   Calibration: {hp.calib_seconds}s")
    print(f"   UP Offset: {hp.up_offset}")
    print(f"   DOWN Offset: {hp.down_offset}")
    print("="*70 + "\n")

    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=hp.mp_model_complexity,
        min_detection_confidence=hp.mp_min_detection_confidence,
        min_tracking_confidence=hp.mp_min_tracking_confidence
    )

    # Setup transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((hp.img_size, hp.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Open webcam
    cap = cv2.VideoCapture(hp.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open camera {hp.camera_index}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # State machine
    state = "CALIBRATING"
    rep_frames_rgb = []
    went_down = False
    went_up = False

    last_result = None
    last_result_time = 0.0

    calib_start = time.time()
    calib_hips = []

    stand_hip = None
    UP_THRESHOLD = None
    DOWN_THRESHOLD = None

    status_text = "Calibrating. Stand still..."
    
    # FPS tracking
    fps_start = time.time()
    frame_count = 0
    current_fps = 0.0

    print("🎬 Starting camera feed...")
    print("   Press 'q' to quit")
    print("   Press 'r' to recalibrate\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = pose.process(rgb)

            # Extract hip position
            hip_y = None
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                if lh.visibility > 0.5 and rh.visibility > 0.5:
                    hip_y = float((lh.y + rh.y) / 2.0)

            now = time.time()
            
            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                current_fps = 30 / (now - fps_start)
                fps_start = now

            # Display result if holding
            if last_result is not None and (now - last_result_time) < hp.result_hold_sec:
                label = last_result["label"]
                conf = last_result["conf"]
                color = hp.colors.get(label.lower(), (255, 255, 255))

                cv2.rectangle(frame, (20, 20), (850, 130), (0, 0, 0), -1)
                cv2.putText(frame, f"Rep Result: {label}", (35, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                cv2.putText(frame, f"Confidence: {conf:.1f}%", (35, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                
                # FPS display
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue
            else:
                if last_result is not None:
                    last_result = None
                    state = "READY"
                    rep_frames_rgb = []
                    went_down = False
                    went_up = False
                    status_text = "Ready: Do ONE squat rep"

            # Check for pose detection
            if hip_y is None:
                status_text = "⚠️  No pose detected"

            # Debug info
            if hip_y is not None:
                cv2.putText(frame, f"hip_y={hip_y:.3f}", (35, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # CALIBRATION STATE
            if state == "CALIBRATING":
                if hip_y is not None:
                    calib_hips.append(hip_y)

                elapsed = now - calib_start
                remaining = max(0, hp.calib_seconds - elapsed)
                status_text = f"⏱️  Calibrating... {remaining:.1f}s"

                if elapsed >= hp.calib_seconds and len(calib_hips) >= hp.calib_min_samples:
                    stand_hip = float(np.median(calib_hips))
                    UP_THRESHOLD = stand_hip + hp.up_offset
                    DOWN_THRESHOLD = stand_hip + hp.down_offset

                    state = "READY"
                    status_text = "✅ Ready: Do ONE squat rep"
                    print(f"✅ Calibration complete!")
                    print(f"   Standing hip: {stand_hip:.3f}")
                    print(f"   UP threshold: {UP_THRESHOLD:.3f}")
                    print(f"   DOWN threshold: {DOWN_THRESHOLD:.3f}\n")

            # READY/RECORDING STATE
            if state in ["READY", "RECORDING"] and hip_y is not None and UP_THRESHOLD is not None:
                # Detect going down
                if hip_y > DOWN_THRESHOLD:
                    if not went_down:
                        print("⬇️  Going down...")
                    went_down = True
                    state = "RECORDING"

                # Record frames
                if state == "RECORDING":
                    rep_frames_rgb.append(rgb)
                    status_text = f"🔴 Recording... ({len(rep_frames_rgb)} frames)"

                # Detect coming back up
                if went_down and hip_y < UP_THRESHOLD:
                    if not went_up:
                        print("⬆️  Coming up...")
                    went_up = True

                # Complete rep
                if went_down and went_up:
                    state = "PROCESSING"
                    print(f"✅ Rep completed! Processing {len(rep_frames_rgb)} frames...")

            # PROCESSING STATE
            if state == "PROCESSING":
                status_text = "⚙️  Processing..."
                
                sampled = sample_uniform(rep_frames_rgb, hp.num_frames)

                if len(sampled) != hp.num_frames:
                    print(f"⚠️  Incomplete rep (only {len(sampled)} frames)")
                    last_result = {
                        "label": "Uncomplete",
                        "conf": 0.0
                    }
                    last_result_time = time.time()
                else:
                    original_stream = []
                    mediapipe_stream = []

                    for fr in sampled:
                        original_stream.append(transform(fr))

                        r = pose.process(fr)
                        drawn = fr.copy()
                        if r.pose_landmarks:
                            mp_draw.draw_landmarks(drawn, r.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        mediapipe_stream.append(transform(drawn))

                    orig_t = torch.stack(original_stream).unsqueeze(0).to(device)
                    mp_t = torch.stack(mediapipe_stream).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = model(orig_t, mp_t)
                        probs = torch.softmax(out, dim=1)
                        conf, pred = torch.max(probs, 1)

                    label_idx = pred.item()
                    label = hp.class_names[label_idx].title()
                    confv = float(conf.item() * 100.0)

                    print(f"🎯 Result: {label} ({confv:.1f}%)\n")

                    last_result = {"label": label, "conf": confv}
                    last_result_time = time.time()

                state = "DONE"

            # Draw UI
            cv2.rectangle(frame, (20, 20), (1000, 130), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (35, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

            if stand_hip is not None:
                cv2.putText(
                    frame,
                    f"Stand={stand_hip:.3f}  UP<{UP_THRESHOLD:.3f}  DOWN>{DOWN_THRESHOLD:.3f}",
                    (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
                )
            
            # FPS display
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):  # Recalibrate
                print("\n🔄 Recalibrating...")
                state = "CALIBRATING"
                calib_start = time.time()
                calib_hips = []
                stand_hip = None
                UP_THRESHOLD = None
                DOWN_THRESHOLD = None
                rep_frames_rgb = []
                went_down = False
                went_up = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n👋 Camera closed. Goodbye!")


if __name__ == "__main__":
    main()