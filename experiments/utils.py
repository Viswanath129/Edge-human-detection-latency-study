import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, num_frames=50, half=False):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    model = YOLO(model_path)

    # Check for half precision support (FP16)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA is not available. Falling back to FP32.")

    # Try to open webcam unless FORCE_SYNTHETIC is set
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found or failed to open. Using synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    print(f"Warming up {model_path} at {imgsz}x{imgsz} (half={actual_half})...")
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    print(f"Benchmarking {num_frames} frames...")
    start_time = time.perf_counter()

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    total_time = end_time - start_time
    fps = len(latencies) / total_time

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates the benchmark result in summary.csv.
    """
    # Absolute path to summary.csv relative to this file
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
            # Check if columns exist (for migration)
            required_cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
            if not all(col in df.columns for col in required_cols):
                print("Old summary schema detected. Re-initializing.")
                df = pd.DataFrame(columns=required_cols)
        except Exception:
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
    else:
        df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

    # Update if exists, else append
    match = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
    if match.any():
        idx = df.index[match][0]
        for key, val in new_data.items():
            df.at[idx, key] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
