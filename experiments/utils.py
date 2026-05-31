import cv2
import time
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=20, force_synthetic=False):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (Average FPS, Average Latency in ms).
    """
    model = YOLO(model_name)

    # Check for webcam or force synthetic
    cap = None
    if not force_synthetic and os.environ.get("FORCE_SYNTHETIC") != "true":
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = None
        except Exception:
            cap = None

    # Warmup
    print(f"Warming up {model_name} (imgsz={imgsz}, half={half})...")
    warmup_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(warmup_frame, imgsz=imgsz, half=half, verbose=False)

    latencies = []

    print(f"Benchmarking {model_name} for {num_frames} frames...")
    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        # Standardize timing using perf_counter for high precision
        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    # FPS calculation based on inference-only latency to avoid I/O noise
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmarking result in the summary CSV.
    """
    # Use absolute path resolution for consistent result saving
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../results/tables/summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Ensure columns match expected schema
            if not all(col in df.columns for col in cols):
                df = pd.DataFrame(columns=cols)
        except Exception:
            df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)

    new_entry = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    # Identify existing entries to perform updates instead of duplicate appends
    mask = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)

    if mask.any():
        for col, val in new_entry.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
