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
    Returns (avg_latency, fps, actual_half)
    """
    model = YOLO(model_name)

    # Check for CUDA and half precision support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_half = half and device == 'cuda'

    if actual_half:
        model.to(device).half()
    else:
        model.to(device)

    # Setup frame source
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
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    latencies = []

    # Generate synthetic frames if needed to avoid overhead in the loop
    if not cap:
        synthetic_frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(num_frames)]

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = synthetic_frames[i]

        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    # Technical FPS: Strictly based on inference latency
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark result in results/tables/summary.csv
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    summary_path = os.path.join(project_root, "results", "tables", "summary.csv")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
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
            # Check for Model and Precision columns (schema migration)
            if "Model" not in df.columns:
                df["Model"] = "yolov8n.pt" # Default for old entries
            if "Precision" not in df.columns:
                df["Precision"] = "FP32" # Default for old entries

            # Update if exists, otherwise append
            mask = (df["Resolution"] == new_entry["Resolution"]) & \
                   (df["Model"] == new_entry["Model"]) & \
                   (df["Precision"] == new_entry["Precision"])

            if mask.any():
                df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = \
                    [new_entry["Average_FPS"], new_entry["Average_Latency_ms"], new_entry["Observation"]]
            else:
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
