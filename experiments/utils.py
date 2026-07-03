import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency_ms, fps, actual_half_used)
    """
    model = YOLO(model_name)

    # Check for CUDA and FP16 support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_half = half and device == 'cuda'

    if actual_half:
        model.to(device).half()
    else:
        model.to(device)

    # Try to open webcam, fallback to synthetic if not available
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False, half=actual_half)

        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (end_time - start_time)

    return avg_latency, fps, actual_half

def save_summary(resolution, fps, latency, observation, model_name, precision):
    """
    Saves or updates benchmark results in results/tables/summary.csv
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "results", "tables", "summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Schema migration: check if Model and Precision columns exist
        if "Model" not in df.columns or "Precision" not in df.columns:
            # Backfill legacy data if possible
            if "Model" not in df.columns:
                df["Model"] = "yolov8n"
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"

        # Check if entry already exists
        match = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)
        if match.any():
            idx = df.index[match][0]
            for col, val in new_data.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
