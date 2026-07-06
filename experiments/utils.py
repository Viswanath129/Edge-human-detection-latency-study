import cv2
import time
import os
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency, fps, actual_half).
    """
    # Load model
    model = YOLO(model_name)

    # Check if half precision is supported (CUDA only)
    actual_half = half and torch.cuda.is_available()

    # Try to open webcam, fallback to synthetic
    cap = cv2.VideoCapture(0)
    use_synthetic = not cap.isOpened() or os.environ.get("FORCE_SYNTHETIC") == "true"

    if use_synthetic:
        print(f"Using synthetic frames for {model_name} at {imgsz}...")
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam, falling back to synthetic.")
            use_synthetic = True
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    latencies = []

    # Warmup
    for _ in range(5):
        try:
            _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError as e:
            if "half" in str(e).lower() and actual_half:
                print("FP16 not supported on this device, falling back to FP32 for warmup.")
                _ = model(frame, imgsz=imgsz, half=False, verbose=False)
            else:
                raise e

    start_time = time.perf_counter()
    for _ in range(num_frames):
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        try:
            _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError as e:
            if "half" in str(e).lower() and actual_half:
                actual_half = False
                _ = model(frame, imgsz=imgsz, half=False, verbose=False)
            else:
                raise e
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    end_time = time.perf_counter()

    if not use_synthetic:
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
    csv_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = {
        "Resolution": [resolution],
        "Model": [model_name],
        "Precision": [precision],
        "Average_FPS": [round(fps, 2)],
        "Average_Latency_ms": [round(latency, 2)],
        "Observation": [observation]
    }
    new_df = pd.DataFrame(new_data)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Schema migration if needed
        if "Model" not in df.columns:
            df["Model"] = "yolov8n" # default
        if "Precision" not in df.columns:
            df["Precision"] = "FP32" # default

        # Check for existing entry to update
        mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
        else:
            df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
