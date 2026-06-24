import cv2
import numpy as np
import time
import torch
import os
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, resolution=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Falls back to synthetic frames if no webcam is available.
    """
    model = YOLO(model_path)

    actual_half = half
    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Force synthetic frames if requested via environment variable or if webcam fails
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    use_synthetic = force_synthetic

    if not use_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Using synthetic frames.")
            use_synthetic = True

    latencies = []

    # Warmup phase
    for _ in range(5):
        if use_synthetic:
            frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (resolution, resolution))
        model(frame, imgsz=resolution, half=actual_half, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_frames):
        if use_synthetic:
            frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (resolution, resolution))

        t0 = time.perf_counter()
        model(frame, imgsz=resolution, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    if cap is not None:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (end_time - start_time)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmarking results in the summary CSV.
    """
    # Use absolute path to ensure it works from any script in experiments/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(base_dir, '../results/tables/summary.csv'))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])
    else:
        df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])

    # Standardize columns for legacy data
    if 'Model' not in df.columns:
        df['Model'] = 'yolov8n'
    if 'Precision' not in df.columns:
        df['Precision'] = 'FP32'

    new_row = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    # Check for existing entry to update
    mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
    if mask.any():
        idx = df.index[mask][0]
        for col, val in new_row.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
