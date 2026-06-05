import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=20, warmup_frames=5):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns (avg_fps, avg_latency_ms).
    """
    model = YOLO(model_path)

    # Check for synthetic force
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup phase
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        model(frame, imgsz=imgsz, half=half, verbose=False)

    latencies = []

    # Benchmarking phase
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return round(avg_fps, 2), round(avg_latency, 2)

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark result in results/tables/summary.csv.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
    csv_path = os.path.abspath(csv_path)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": fps,
        "Average_Latency_ms": latency,
        "Observation": observation
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check if entry already exists (Resolution, Model, Precision)
        match = (df["Resolution"] == resolution) & \
                (df["Model"] == model_name) & \
                (df["Precision"] == precision)

        if match.any():
            idx = df.index[match][0]
            for key, value in new_data.items():
                df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
