import cv2
import time
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency_ms, fps, actual_half)
    """
    # Load model
    model = YOLO(model_name)

    # FP16 safety check
    actual_half = half
    if half:
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            print("CUDA not available. Falling back to FP32 for benchmark.")
            actual_half = False

    # Frame source: synthetic or webcam
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Using synthetic frames.")
            cap = None

    # Warmup
    print(f"Warming up {model_name} (imgsz={imgsz}, half={actual_half})...")
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    # Benchmark loop
    print(f"Benchmarking {model_name} over {num_frames} frames...")
    latencies = []
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False, half=actual_half)
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
    Saves or updates benchmark results in results/tables/summary.csv
    """
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
            # Check for schema compatibility
            required_cols = ["Resolution", "Model", "Precision"]
            if not all(col in df.columns for col in required_cols):
                print("Old summary schema detected. Migrating...")
                df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
        except Exception:
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
    else:
        df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

    # Update or append
    match = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)
    if match.any():
        idx = df.index[match][0]
        for key, value in new_data.items():
            df.at[idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
