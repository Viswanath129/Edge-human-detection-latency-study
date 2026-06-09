import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50, warmup_frames=5):
    """
    Benchmarks a YOLO model and returns average FPS and latency.
    """
    # Load model
    model = YOLO(model_path)

    # Check for half precision support
    if half:
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            print("Warning: FP16 requested but CUDA not available. Falling back to FP32.")
            half = False

    # Initialize video source (webcam or synthetic)
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found or FORCE_SYNTHETIC is true. Using synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=half, verbose=False)

    # Benchmark loop
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark entry in results/tables/summary.csv.
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
            # Check if entry already exists to update it
            mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
            if mask.any():
                for key, value in new_data.items():
                    df.loc[mask, key] = value
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except (KeyError, pd.errors.EmptyDataError):
            # If schema mismatch or empty, start fresh or try to adapt
            # For this research task, starting fresh with new schema is safer
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
