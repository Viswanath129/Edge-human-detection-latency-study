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
    Returns (avg_latency_ms, fps, actual_half)
    """
    model = YOLO(model_name)

    # FP16 safety check
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected, falling back to synthetic frames.")
            cap = None

    # Warmup phase
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark entry in results/tables/summary.csv
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
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
        try:
            df = pd.read_csv(csv_path)
            # Check for legacy schema and migrate if necessary
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Migrating legacy summary.csv schema...")
                # If it's the old schema, it only had Resolution, Average_FPS, Average_Latency_ms, Observation
                # We'll try to preserve what we can but some info might be lost or need defaults
                if "Model" not in df.columns: df["Model"] = "yolov8n.pt"
                if "Precision" not in df.columns: df["Precision"] = "FP32"

            # Identify if entry already exists to update it
            match = (df["Resolution"] == resolution) & \
                    (df["Model"] == model_name) & \
                    (df["Precision"] == precision)

            if match.any():
                idx = df.index[match][0]
                for key, value in new_data.items():
                    df.at[idx, key] = value
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}. Creating new.")
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
