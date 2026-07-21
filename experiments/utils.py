import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name: str, resolution: int, precision: str = "FP32", num_frames: int = 50) -> tuple:
    """
    Benchmarks a YOLO model for latency and FPS.

    Returns:
        tuple: (avg_latency_ms, fps, actual_half)
    """
    # 1. Initialize YOLO Model
    model = YOLO(model_name)

    # 2. Setup precision (FP16 check/fallback)
    requested_half = (precision.upper() == "FP16")
    actual_half = False

    if requested_half:
        if torch.cuda.is_available():
            # Model can be moved to half on CUDA
            model.to('cuda')
            actual_half = True
        else:
            # CPU fallback or warning
            print(f"CUDA is not available. Standard FP32 precision will be used instead of requested FP16.")
            actual_half = False

    # 3. Handle physical webcam vs synthetic frame source
    # prioritize cv2.VideoCapture(0) unless FORCE_SYNTHETIC is set
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    webcam_active = False

    if not force_synthetic:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Let's read one test frame to confirm it works
                ret, frame = cap.read()
                if ret:
                    webcam_active = True
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            print(f"Failed to initialize webcam: {e}")
            cap = None

    # Pre-generate synthetic frames if webcam is unavailable
    synthetic_frames = []
    if not webcam_active:
        print("Using synthetic fallback frames.")
        # Pre-generate synthetic frames OUTSIDE the timed loop to minimize acquisition overhead
        for _ in range(num_frames + 10): # extra frames for safety/warm-up
            # Generate a random dummy image frame (H, W, C)
            synthetic_frames.append(np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8))

    # 4. Warm-up Phase (5 frames)
    # Uses the exact same precision settings as the main loop
    warmup_frames = 5
    for i in range(warmup_frames):
        if webcam_active:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = synthetic_frames[i]

        # Preprocess frame sizing as required
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            frame = cv2.resize(frame, (resolution, resolution))

        # Run warmup inference
        try:
            _ = model(frame, imgsz=resolution, half=actual_half, verbose=False)
        except Exception as e:
            print(f"Warmup inference failed: {e}")

    # 5. Main Inference Loop
    latencies = []
    frame_idx = 0
    start_time = time.time()

    for _ in range(num_frames):
        if webcam_active:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx + warmup_frames < len(synthetic_frames):
                frame = synthetic_frames[frame_idx + warmup_frames]
            else:
                frame = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)

        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            frame = cv2.resize(frame, (resolution, resolution))

        # Perform CUDA synchronization before starting measurement if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        try:
            # Run inference
            _ = model(frame, imgsz=resolution, half=actual_half, verbose=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000  # in ms
            latencies.append(latency)
        except RuntimeError as e:
            # Handle standard precision or catch RuntimeError on CPU to prevent benchmarking crashes
            print(f"Inference error encountered: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            break

        frame_idx += 1

    # Release webcam if opened
    if cap is not None:
        cap.release()

    # 6. Metrics Calculation & Hardening
    # Hardened against division-by-zero
    if len(latencies) == 0:
        return (0.0, 0.0, actual_half)

    avg_latency = sum(latencies) / len(latencies)
    # inference-only FPS
    # We want strictly inference-only performance (avoid noise from frame acquisition)
    # Using average latency to get inference-only FPS
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return (avg_latency, fps, actual_half)


def save_summary(resolution_str: str, model_variant: str, precision_str: str, avg_fps: float, avg_latency: float, observation: str):
    """
    Consolidates results into results/tables/summary.csv with schema:
    Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation

    Supports updates instead of duplicate appends and legacy schema migration.
    """
    # Standardize model names to "YOLOv8" prefix with a lowercase suffix (e.g., 'yolov8n' -> 'YOLOv8n')
    # Let's normalize model_variant
    normalized_model = model_variant
    if model_variant.lower().startswith('yolov8'):
        suffix = model_variant.lower().replace('yolov8', '')
        normalized_model = f"YOLOv8{suffix}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_dir = os.path.join(project_root, 'results', 'tables')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'summary.csv')

    # Read existing or create new dataframe
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)

            # Legacy Schema Migration Check: detect missing 'Model' or 'Precision' columns and populate them
            cols_to_add = {}
            if 'Model' not in df.columns:
                cols_to_add['Model'] = 'YOLOv8n'
            if 'Precision' not in df.columns:
                cols_to_add['Precision'] = 'FP32'

            if cols_to_add:
                for col, val in cols_to_add.items():
                    df[col] = val
                print(f"Legacy schema migration completed. Added missing columns: {list(cols_to_add.keys())}")
        except Exception as e:
            print(f"Error reading existing summary.csv: {e}. Re-initializing.")
            df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])
    else:
        df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])

    # Standardize column list
    required_cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # Clean/standardize types
    df['Resolution'] = df['Resolution'].astype(str)
    df['Model'] = df['Model'].astype(str)
    df['Precision'] = df['Precision'].astype(str)

    # Create the new row
    new_row = {
        'Resolution': str(resolution_str),
        'Model': str(normalized_model),
        'Precision': str(precision_str),
        'Average_FPS': float(avg_fps),
        'Average_Latency_ms': float(avg_latency),
        'Observation': str(observation)
    }

    # Find matching entry by Resolution, Model, and Precision to update instead of duplicating
    match_idx = df[(df['Resolution'] == resolution_str) &
                   (df['Model'] == normalized_model) &
                   (df['Precision'] == precision_str)].index

    if not match_idx.empty:
        # Update existing
        for col, val in new_row.items():
            df.loc[match_idx, col] = val
        print(f"Updated existing entry in summary.csv for {resolution_str} / {normalized_model} / {precision_str}")
    else:
        # Append new
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Appended new entry to summary.csv for {resolution_str} / {normalized_model} / {precision_str}")

    # Reorder columns to match standardized schema
    df = df[required_cols]

    # Write back to summary.csv
    df.to_csv(csv_path, index=False)
    print(f"Summary saved successfully to {csv_path}")
