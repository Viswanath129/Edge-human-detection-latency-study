import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency_ms, fps, actual_half).
    """
    model = YOLO(model_path)

    # Check if half precision is supported (CUDA only for most YOLO implementations)
    actual_half = half and torch.cuda.is_available()

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Synthetic frame fallback
    if cap is None:
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    # Benchmark loop
    start_time = time.perf_counter()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            # For synthetic, we use the pre-generated frame to avoid timing overhead of generation
            pass

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

def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates the benchmark result in the central summary.csv.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "..", "results", "tables", "summary.csv"))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
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
            # Schema migration if needed
            if "Model" not in df.columns:
                df["Model"] = "YOLOv8n"  # Assume YOLOv8n for legacy data
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"  # Assume FP32 for legacy data
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

        # Check for existing entry to update
        match = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)
        if match.any():
            df.loc[match, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [new_entry["Average_FPS"], new_entry["Average_Latency_ms"], new_entry["Observation"]]
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
