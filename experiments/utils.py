import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name="yolov8n.pt", imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns: (avg_latency_ms, fps, actual_half)
    """
    # Load model
    model = YOLO(model_name)

    # Check for CUDA and half precision support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if half and device == "cpu":
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        half = False

    model.to(device)

    # Determine frame source
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, falling back to synthetic frames.")
            cap = None

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=half, verbose=False)

    # Benchmarking loop
    latencies = []

    # Pre-generate synthetic frames to minimize overhead if in synthetic mode
    if not cap:
        synthetic_frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(num_frames)]

    t_start = time.perf_counter()
    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = synthetic_frames[i]

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    t_end = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (t_end - t_start)

    return avg_latency, fps, half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmarking results in results/tables/summary.csv
    """
    # Normalize model name for consistent display
    if model_name.startswith("yolov8"):
        model_display = "YOLOv8" + model_name[6]
    else:
        model_display = model_name

    data = {
        "Resolution": [resolution],
        "Model": [model_display],
        "Precision": [precision],
        "Average_FPS": [round(fps, 2)],
        "Average_Latency_ms": [round(latency, 2)],
        "Observation": [observation]
    }

    new_df = pd.DataFrame(data)

    # Use absolute path relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    table_dir = os.path.join(base_dir, "results", "tables")
    os.makedirs(table_dir, exist_ok=True)
    summary_path = os.path.join(table_dir, "summary.csv")

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Ensure 'Model' and 'Precision' columns exist for backward compatibility
        if "Model" not in df.columns:
            df["Model"] = "YOLOv8n"
        if "Precision" not in df.columns:
            df["Precision"] = "FP32"

        # Check if entry already exists to update it
        mask = (df["Resolution"] == resolution) & \
               (df["Model"] == model_display) & \
               (df["Precision"] == precision)

        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = \
                [round(fps, 2), round(latency, 2), observation]
        else:
            df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
