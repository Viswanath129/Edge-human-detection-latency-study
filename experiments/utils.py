import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference performance.
    Returns (avg_latency_ms, fps, actual_half_used)
    """
    model = YOLO(model_name)

    # Check if half precision is supported (requires CUDA)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")

    # Force synthetic frames for benchmarking to avoid I/O overhead and support headless
    # We use pre-generated frames to minimize overhead
    frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(5)]

    # Warmup
    print(f"Warming up {model_name} (imgsz={imgsz}, half={actual_half})...")
    for _ in range(5):
        model(frames[0], imgsz=imgsz, half=actual_half, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    print(f"Benchmarking {model_name} for {num_frames} frames...")

    start_time = time.perf_counter()
    for i in range(num_frames):
        frame = frames[i % 5]
        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_time = end_time - start_time
    fps = num_frames / total_time if total_time > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark results in the centralized summary CSV.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../results/tables/summary.csv")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Normalize model name for consistency
    model_id = "YOLOv8" + model_name.lower().replace("yolov8", "")

    new_data = {
        "Resolution": f"{resolution}x{resolution}" if isinstance(resolution, int) else resolution,
        "Model": model_id,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Ensure all required columns exist (handle legacy schema)
        if "Model" not in df.columns:
            df["Model"] = "YOLOv8n" # Default legacy
        if "Precision" not in df.columns:
            df["Precision"] = "FP32" # Default legacy

        # Normalize existing entries for matching
        df["Model"] = df["Model"].apply(lambda x: "YOLOv8" + x.lower().replace("yolov8", ""))

        # Try to update existing entry
        mask = (df["Resolution"] == new_data["Resolution"]) & \
               (df["Model"] == new_data["Model"]) & \
               (df["Precision"] == new_data["Precision"])

        if mask.any():
            for col, val in new_data.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    # Maintain column order
    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    df = df[cols]

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
