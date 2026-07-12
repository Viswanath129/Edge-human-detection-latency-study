import time
import torch
import numpy as np
import pandas as pd
import os
import cv2
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency, fps, actual_half).
    """
    model = YOLO(model_name)

    # Check if half precision is possible (only on CUDA)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: Half precision requested but CUDA not available. Falling back to FP32.")

    # Warmup
    print(f"Warming up {model_name} at {imgsz}x{imgsz} (half={actual_half})...")
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy_frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    print(f"Benchmarking {model_name}...")

    # Determine frame source
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Could not open webcam. Falling back to synthetic frames.")
            cap = None

    frames = []
    if cap:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (imgsz, imgsz)))
        cap.release()

    # Fallback to synthetic if webcam failed or was skipped
    if len(frames) < num_frames:
        needed = num_frames - len(frames)
        synthetic_frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(needed)]
        frames.extend(synthetic_frames)

    start_time = time.perf_counter()
    for frame in frames:
        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    end_time = time.perf_counter()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fps = len(latencies) / (end_time - start_time) if (end_time - start_time) > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates an entry in results/tables/summary.csv
    """
    # Normalize model name
    if model_name.lower().startswith('yolov8'):
        model_name = 'YOLOv8' + model_name[6:]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, '../results/tables/summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": [resolution],
        "Model": [model_name],
        "Precision": [precision],
        "Average_FPS": [round(fps, 2)],
        "Average_Latency_ms": [round(latency, 2)],
        "Observation": [observation]
    }
    new_df = pd.DataFrame(new_data)

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Ensure schema consistency for legacy files
        if 'Model' not in df.columns:
            df['Model'] = 'YOLOv8n'
        if 'Precision' not in df.columns:
            df['Precision'] = 'FP32'

        # Update existing or append
        mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
        else:
            df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")
