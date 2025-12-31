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


# =========================
# CONFIG (from config.py)
# =========================
MODEL_PATH = str(config.SQUATS_MODEL)
NUM_FRAMES = 40
IMG_SIZE = config.IMG_SIZE

WINDOW_NAME = "Squat - One Rep Judge (Real Time)"

CALIB_SECONDS = 5.0
CALIB_MIN_SAMPLES = 15

UP_OFFSET = 0.02
DOWN_OFFSET = 0.08

RESULT_HOLD_SEC = 2.0

CLASSES = ["Correct", "Fast", "Uncomplete", "Wrong Position"]
COLORS = {
    "Correct": (0, 255, 0),
    "Fast": (0, 255, 255),
    "Uncomplete": (0, 165, 255),
    "Wrong Position": (0, 0, 255),
}


def sample_uniform(frames, n=16):
    if len(frames) == 0:
        return []
    if len(frames) == n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def safe_load_state_dict(model, checkpoint, device):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoStreamSquatClassifier(num_classes=4).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    safe_load_state_dict(model, ckpt, device)
    print("✅ Model loaded:", MODEL_PATH)
    print("✅ Device:", device)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam. Try changing camera index (0/1/2).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

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

    status_text = "Calibrating. Stand still (2s)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        hip_y = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            if lh.visibility > 0.5 and rh.visibility > 0.5:
                hip_y = float((lh.y + rh.y) / 2.0)

        now = time.time()

        if last_result is not None and (now - last_result_time) < RESULT_HOLD_SEC:
            label = last_result["label"]
            conf = last_result["conf"]
            color = COLORS.get(label, (255, 255, 255))

            cv2.rectangle(frame, (20, 20), (820, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Rep Result: {label}", (35, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            cv2.putText(frame, f"Confidence: {conf:.1f}%", (35, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue
        else:
            if last_result is not None:
                last_result = None
                state = "READY"
                rep_frames_rgb = []
                went_down = False
                went_up = False
                status_text = "Ready: Do ONE squat rep (Down -> Up)"

        if hip_y is None:
            status_text = "No pose detected. Step back (hips+knees visible)"

        if hip_y is not None:
            cv2.putText(frame, f"hip_y={hip_y:.3f}", (35, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if state == "CALIBRATING":
            if hip_y is not None:
                calib_hips.append(hip_y)

            elapsed = now - calib_start
            status_text = f"Calibrating. Stand still ({max(0, CALIB_SECONDS-elapsed):.1f}s)"

            if elapsed >= CALIB_SECONDS and len(calib_hips) >= CALIB_MIN_SAMPLES:
                stand_hip = float(np.median(calib_hips))
                UP_THRESHOLD = stand_hip + UP_OFFSET
                DOWN_THRESHOLD = stand_hip + DOWN_OFFSET

                state = "READY"
                status_text = "Ready: Do ONE squat rep (Down -> Up)"
                print(f"✅ Calibration done. stand_hip={stand_hip:.3f}, UP<{UP_THRESHOLD:.3f}, DOWN>{DOWN_THRESHOLD:.3f}")

        if state in ["READY", "RECORDING"] and hip_y is not None and UP_THRESHOLD is not None and DOWN_THRESHOLD is not None:
            if hip_y > DOWN_THRESHOLD:
                went_down = True
                state = "RECORDING"

            if state == "RECORDING":
                rep_frames_rgb.append(rgb)

            if went_down and hip_y < UP_THRESHOLD:
                went_up = True

            if went_down and went_up:
                state = "DONE"

        if state == "DONE":
            sampled = sample_uniform(rep_frames_rgb, NUM_FRAMES)

            if len(sampled) != NUM_FRAMES:
                last_result = {"label": "Uncomplete", "conf": 0.0}
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

                label = CLASSES[pred.item()]
                confv = float(conf.item() * 100.0)

                last_result = {"label": label, "conf": confv}
                last_result_time = time.time()

            status_text = "Done!"

        cv2.rectangle(frame, (20, 20), (980, 120), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (35, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if stand_hip is not None:
            cv2.putText(frame, f"stand={stand_hip:.3f}  UP<{UP_THRESHOLD:.3f}  DOWN>{DOWN_THRESHOLD:.3f}",
                        (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
