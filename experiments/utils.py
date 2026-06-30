import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for a set number of frames.
    Returns: (avg_latency_ms, fps, actual_half)
    """
    model = YOLO(model_name)

    # Handle FP16 check
    actual_half = half
    if half:
        if not torch.cuda.is_available():
            print("Warning: FP16 requested but CUDA not available. Falling back to FP32.")
            actual_half = False

    # Check for webcam or force synthetic
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Using synthetic frames.")
            cap = None

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    total_time = end_time - start_time
    fps = len(latencies) / total_time

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark entry in the central summary.csv.
    """
    # Use absolute path resolution relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "../results/tables/summary.csv"))

    # Ensure directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Schema migration check
            if 'Model' not in df.columns or 'Precision' not in df.columns:
                print("Old schema detected. Migrating summary.csv.")
                df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
        except Exception:
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
    else:
        df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

    # Update or append
    mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
    if not df[mask].empty:
        idx = df[mask].index[0]
        for col, val in new_data.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
