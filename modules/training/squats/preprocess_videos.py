"""
Pre-process all videos with MediaPipe once to avoid runtime issues.
This creates processed video files that can be loaded directly during training.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from tqdm import tqdm
import pickle

import modules.config as config

# Lazy import MediaPipe to avoid early loading issues
mp = None
mp_pose = None
mp_draw = None


def init_mediapipe():
    """Initialize MediaPipe lazily"""
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
            print("\n🔧 Try fixing with:")
            print("   pip uninstall protobuf tensorflow mediapipe -y")
            print("   pip install protobuf==3.20.3")
            print("   pip install mediapipe==0.10.9")
            sys.exit(1)


def process_single_video(video_path, num_frames=40):
    """
    Extract frames and apply MediaPipe to a single video
    Returns: (original_frames, mediapipe_frames) as numpy arrays
    """
    # Initialize MediaPipe if not already done
    if mp_pose is None:
        init_mediapipe()
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        # Read video
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total <= 0:
            cap.release()
            return None, None
        
        # Sample frames uniformly
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        original_frames = []
        mediapipe_frames = []
        
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Original frame
                original_frames.append(rgb)
                
                # MediaPipe processed frame
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
        
        if len(original_frames) == num_frames:
            return np.array(original_frames), np.array(mediapipe_frames)
        else:
            return None, None


def preprocess_all_videos():
    """Process all videos and save to disk"""
    
    print("="*70)
    print("🎬 PRE-PROCESSING VIDEOS WITH MEDIAPIPE")
    print("="*70)
    
    # Initialize MediaPipe first
    print("\n🔄 Initializing MediaPipe...")
    init_mediapipe()
    
    hp = config.SQUAT_HP
    
    # Create output directory
    processed_dir = config.DATA_DIR.parent / "squats_processed"
    processed_dir.mkdir(exist_ok=True)
    
    print(f"\n📂 Input:  {config.DATA_DIR}")
    print(f"📂 Output: {processed_dir}\n")
    
    all_videos = []
    for cls in hp.class_names:
        cls_dir = config.DATA_DIR / cls
        if cls_dir.exists():
            for video_file in cls_dir.glob("*"):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    all_videos.append((cls, video_file))
    
    print(f"📊 Found {len(all_videos)} videos to process\n")
    
    # Process each video
    processed_data = []
    failed_videos = []
    
    for cls, video_path in tqdm(all_videos, desc="Processing videos"):
        try:
            original, mediapipe = process_single_video(video_path, hp.num_frames)
            
            if original is not None and mediapipe is not None:
                # Save as compressed numpy file
                output_name = f"{cls}_{video_path.stem}.npz"
                output_path = processed_dir / output_name
                
                np.savez_compressed(
                    output_path,
                    original=original,
                    mediapipe=mediapipe,
                    label=hp.class_names.index(cls),
                    video_name=str(video_path.name)
                )
                
                processed_data.append({
                    'class': cls,
                    'video': video_path.name,
                    'output': output_name
                })
            else:
                failed_videos.append(str(video_path))
                
        except Exception as e:
            print(f"\n❌ Error processing {video_path}: {e}")
            failed_videos.append(str(video_path))
    
    # Save metadata
    metadata_path = processed_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'processed_data': processed_data,
            'failed_videos': failed_videos,
            'hyperparameters': hp,
            'num_videos': len(processed_data),
            'class_names': hp.class_names
        }, f)
    
    print("\n" + "="*70)
    print("✅ PRE-PROCESSING COMPLETE!")
    print("="*70)
    print(f"✅ Successfully processed: {len(processed_data)} videos")
    if failed_videos:
        print(f"❌ Failed to process: {len(failed_videos)} videos")
        for vid in failed_videos[:5]:  # Show first 5
            print(f"   - {vid}")
    print(f"\n📂 Output saved to: {processed_dir}")
    print("="*70)


if __name__ == "__main__":
    preprocess_all_videos()
    