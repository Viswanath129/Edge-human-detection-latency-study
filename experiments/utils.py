import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns (avg_latency_ms, fps, actual_half).
    """
    model = YOLO(model_name)

    # Check if FP16 is supported (CUDA required for significant speedup)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Falling back to synthetic frames.")
            cap = None

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        # Inference only timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmarking results in results/tables/summary.csv.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, "results", "tables", "summary.csv")

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
            # Check for schema migration
            if "Model" not in df.columns or "Precision" not in df.columns:
                raise ValueError("Old schema detected")

            # Update if exists, else append
            match = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)
            if match.any():
                df.loc[match, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [new_data["Average_FPS"], new_data["Average_Latency_ms"], new_data["Observation"]]
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except (pd.errors.EmptyDataError, ValueError):
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
