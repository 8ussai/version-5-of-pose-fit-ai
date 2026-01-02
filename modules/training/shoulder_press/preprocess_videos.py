import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import modules.config as config

# Lazy import mediapipe
mp = None
mp_pose = None
mp_draw = None


def init_mediapipe():
    global mp, mp_pose, mp_draw
    if mp is None:
        try:
            import mediapipe as mp_module
            mp = mp_module
            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"❌ Error importing MediaPipe: {e}")
            print("\n🔧 Fix suggestion:")
            print("   pip uninstall protobuf tensorflow mediapipe -y")
            print("   pip install protobuf==3.20.3")
            print("   pip install mediapipe==0.10.9")
            sys.exit(1)


def process_single_video(video_path: Path, num_frames: int, mp_model_complexity: int = 1):
    if mp_pose is None:
        init_mediapipe()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(mp_model_complexity),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None, None

        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        original_frames, mediapipe_frames = [], []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frames.append(rgb)

            results = pose.process(rgb)
            mp_frame = rgb.copy()
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    mp_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            mediapipe_frames.append(mp_frame)

        cap.release()

        if len(original_frames) == num_frames and len(mediapipe_frames) == num_frames:
            return np.array(original_frames), np.array(mediapipe_frames)
        return None, None


def preprocess_all_videos():
    config.ensure_dirs()
    hp = config.SHOULDER_PRESS_HP

    # Shoulder Press paths
    data_dir = config.SHOULDER_PRESS_DATA_DIR
    processed_dir = config.SHOULDER_PRESS_PROCESSED_DIR
    processed_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("🎬 PRE-PROCESSING SHOULDER PRESS VIDEOS -> NPZ")
    print("=" * 70)
    print(f"📂 Input:  {data_dir}")
    print(f"📂 Output: {processed_dir}\n")

    all_videos = []
    for cls in hp.class_names:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            for video_file in cls_dir.glob("*"):
                if video_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    all_videos.append((cls, video_file))

    print(f"📊 Found {len(all_videos)} videos\n")

    failed = 0
    for cls, video_path in tqdm(all_videos, desc="Processing"):
        original, mediapipe = process_single_video(
            video_path=video_path,
            num_frames=hp.num_frames,
            mp_model_complexity=hp.mp_model_complexity
        )

        if original is None or mediapipe is None:
            failed += 1
            continue

        out_name = f"{cls}_{video_path.stem}.npz"
        out_path = processed_dir / out_name

        np.savez_compressed(
            out_path,
            original=original,          # (T,H,W,C) uint8
            mediapipe=mediapipe,        # (T,H,W,C) uint8
            label=hp.class_names.index(cls),
            video_name=video_path.name
        )

    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print(f"✅ Saved npz files to: {processed_dir}")
    if failed:
        print(f"⚠️ Failed videos: {failed}")


if __name__ == "__main__":
    preprocess_all_videos()