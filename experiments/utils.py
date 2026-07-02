import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    model = YOLO(model_name)

    # Force CPU if CUDA not available but half requested
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Detect frame source
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    warmup_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(warmup_frame, imgsz=imgsz, half=actual_half, verbose=False)

    start_time = time.perf_counter()

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (end_time - start_time)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Updates the central summary.csv file with benchmark results.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, "results", "tables", "summary.csv")

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
        except Exception:
            df = pd.DataFrame(columns=new_data.keys())

        # Ensure all columns exist for merge
        for col in new_data.keys():
            if col not in df.columns:
                df[col] = "N/A"

        # Update or append
        match = (df.get("Resolution") == resolution) & \
                (df.get("Model") == model_name) & \
                (df.get("Precision") == precision)
        if match.any():
            for key, value in new_data.items():
                df.loc[match, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
