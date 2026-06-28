import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50, warmup_frames=5):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns (avg_latency_ms, fps, actual_half)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    # If half is requested but no CUDA, fallback or at least note it.
    actual_half = half and device == 'cuda'

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        model(frame, imgsz=imgsz, half=actual_half, device=device, verbose=False)

    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, device=device, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    total_time = time.perf_counter() - start_time

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / total_time # Using total loop time for FPS, or just 1000/avg_latency?
    # Usually FPS includes everything in the loop.
    # Memory says "FPS is calculated based strictly on inference latency (inference-only FPS)"
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the summary in results/tables/summary.csv
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    summary_path = os.path.join(project_root, "results", "tables", "summary.csv")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
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
            # Check for schema compatibility
            required_cols = ["Resolution", "Model", "Precision"]
            if not all(col in df.columns for col in required_cols):
                # Legacy schema, just overwrite or handle carefully.
                # Given the task, we want the new schema.
                df = pd.DataFrame([new_entry])
            else:
                # Update if exists
                match = (df['Resolution'] == resolution) & \
                        (df['Model'] == model_name) & \
                        (df['Precision'] == precision)

                if match.any():
                    df.loc[match, ["Average_FPS", "Average_Latency_ms", "Observation"]] = \
                        [new_entry["Average_FPS"], new_entry["Average_Latency_ms"], new_entry["Observation"]]
                else:
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")
