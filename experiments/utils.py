import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency_ms, fps, actual_half).
    """
    # Load model
    model = YOLO(model_path)

    # Check if half precision is possible
    actual_half = half
    if half:
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            print("CUDA not available, falling back to FP32 for benchmark.")
            actual_half = False

    # Try to open webcam unless forced synthetic
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
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    # Benchmark loop
    latencies = []

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                # If webcam fails mid-stream, fallback
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
    Saves or updates the summary.csv file.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/tables/summary.csv")
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
        # Check if the entry already exists to update it
        mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if mask.any():
            for col, val in new_data.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
