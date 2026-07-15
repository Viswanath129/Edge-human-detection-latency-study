import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency_ms, fps, actual_half).
    """
    model = YOLO(model_name)

    # Check if half precision is supported (requires CUDA)
    actual_half = half and torch.cuda.is_available()
    if actual_half:
        model.to('cuda').half()

    # Try to open webcam, fallback to synthetic if fails or FORCE_SYNTHETIC is set
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    latencies = []

    # Warmup
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    start_time = time.perf_counter()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # For synthetic, we use the same frame to isolate inference latency
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)
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

def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates the benchmark results in results/tables/summary.csv.
    """
    # Normalize model name for consistency
    model_display = model_name.replace(".pt", "")
    if model_display.startswith("yolov8"):
        model_display = "YOLOv8" + model_display[6:]

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "tables", "summary.csv"))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_display,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Check if columns exist, if not, it's the old schema
        if "Model" not in df.columns:
            df["Model"] = "YOLOv8n"
        if "Precision" not in df.columns:
            df["Precision"] = "FP32"

        # Update existing entry if Resolution, Model, and Precision match
        mask = (df["Resolution"] == resolution) & (df["Model"] == model_display) & (df["Precision"] == precision)
        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [new_data["Average_FPS"], new_data["Average_Latency_ms"], observation]
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    # Reorder columns to match schema
    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    df = df[cols]

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
