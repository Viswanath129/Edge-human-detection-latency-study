import cv2
import time
import os
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency, fps, actual_half)
    """
    model = YOLO(model_path)

    # Check for FP16 support
    actual_half = half
    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Detect frame source
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected, falling back to synthetic frames.")
            cap = None

    # Warmup phase (5 frames)
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        try:
            model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError as e:
            if "half" in str(e).lower():
                print(f"FP16 not supported on this hardware: {e}")
                actual_half = False
                model(frame, imgsz=imgsz, half=actual_half, verbose=False)
            else:
                raise e

    # Benchmarking loop
    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    total_time = end_time - start_time
    fps = len(latencies) / total_time if total_time > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/tables/summary.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Schema migration check
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Old schema detected. Migrating summary.csv...")
                # Re-initialize with new schema if old one is incompatible
                df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

            # Check for existing entry
            match = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)

            if match.any():
                idx = df.index[match][0]
                for col, val in new_data.items():
                    df.at[idx, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary.csv: {e}. Creating new file.")
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(file_path, index=False)
    print(f"Saved results for {model_name} at {resolution} ({precision}) to {file_path}")
