import cv2
import time
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, resolution, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency, fps, actual_half)
    """
    model = YOLO(model_name)

    # Check for FP16 support
    actual_half = half
    if half:
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
            actual_half = False

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, using synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        model(frame, imgsz=resolution, verbose=False)

    start_bench = time.perf_counter()
    processed_count = 0

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=resolution, half=actual_half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        processed_count += 1

    end_bench = time.perf_counter()

    if cap:
        cap.release()

    if processed_count == 0:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / processed_count
    fps = processed_count / (end_bench - start_bench)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmarking result in results/tables/summary.csv.
    """
    summary_path = "results/tables/summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": f"{resolution}x{resolution}",
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Ensure required columns exist for matching (migration)
            if "Model" not in df.columns or "Precision" not in df.columns:
                raise KeyError("Missing schema columns")

            # Update if exists
            mask = (df['Resolution'] == new_data['Resolution']) & \
                   (df['Model'] == new_data['Model']) & \
                   (df['Precision'] == new_data['Precision'])

            if mask.any():
                df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = \
                    [new_data["Average_FPS"], new_data["Average_Latency_ms"], new_data["Observation"]]
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except (pd.errors.EmptyDataError, KeyError):
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
