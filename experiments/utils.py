import os
import time
import numpy as np
import pandas as pd
import torch
import cv2
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference performance.
    Handles headless environments by falling back to synthetic frames.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    # FP16 check - only supported on CUDA/NPU
    actual_half = half and torch.cuda.is_available()

    try:
        if actual_half:
            model.to(device).half()
        else:
            model.to(device)
    except RuntimeError:
        # Fallback if half precision fails on certain devices
        actual_half = False
        model.to(device)

    # Frame source detection
    force_synthetic = os.getenv('FORCE_SYNTHETIC', 'false').lower() == 'true'
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup phase (5 frames)
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    latencies = []

    # Inference loop
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False, half=actual_half)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv.
    Handles schema migration for legacy files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, '../results/tables/summary.csv')
    summary_path = os.path.normpath(summary_path)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Schema migration: ensure all expected columns exist
            required_cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = "Legacy" if col in ['Model', 'Precision'] else None

            # Check for existing entry to update
            mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
            if mask.any():
                for col, val in new_data.items():
                    df.loc[mask, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Summary updated: {model_name} ({precision}) at {resolution}")
