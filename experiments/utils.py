import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    """
    # Load model
    model = YOLO(model_name)

    # Check if half precision is supported (only on CUDA)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")

    # Initialize frame source
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected, falling back to synthetic frames.")
            cap = None

    # Warmup phase (5 frames)
    print(f"Warming up {model_name} (imgsz={imgsz}, half={actual_half})...")
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    # Benchmark loop
    print(f"Benchmarking {num_frames} frames...")
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

def save_summary(res, fps, latency, model_name, precision, observation):
    """
    Saves or updates the summary results in results/tables/summary.csv
    """
    csv_path = "results/tables/summary.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = {
        "Resolution": f"{res}x{res}",
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Schema migration check
        required_cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "N/A"

        # Check for existing entry to update
        match = (df["Resolution"] == new_data["Resolution"]) & \
                (df["Model"] == new_data["Model"]) & \
                (df["Precision"] == new_data["Precision"])

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
