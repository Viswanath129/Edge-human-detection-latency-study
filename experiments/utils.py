import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference-only performance.

    Returns:
        tuple: (avg_latency_ms, fps, actual_half)
    """
    model = YOLO(model_path)

    # Check for CUDA and handle FP16 fallback
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_half = half and device == 'cuda'

    # Force synthetic frames if requested or if no webcam
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Generate synthetic frame if needed
    synthetic_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup phase (5 frames)
    for _ in range(5):
        _ = model(synthetic_frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    try:
        for _ in range(num_frames):
            if cap:
                ret, frame = cap.read()
                if not ret:
                    frame = synthetic_frame
                else:
                    frame = cv2.resize(frame, (imgsz, imgsz))
            else:
                frame = synthetic_frame

            # Measure inference only
            if device == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

            if device == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    finally:
        if cap:
            cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    csv_path = os.path.join(project_root, 'results', 'tables', 'summary.csv')

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Check for existing entry to update
            mask = (df['Resolution'] == resolution) & \
                   (df['Model'] == model_name) & \
                   (df['Precision'] == precision)

            if mask.any():
                for col, val in new_data.items():
                    df.loc[mask, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception:
            # If CSV is malformed or missing columns, overwrite it
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
