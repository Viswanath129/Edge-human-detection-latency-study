import cv2
import time
import numpy as np
import pandas as pd
import os
import torch
from ultralytics import YOLO

def benchmark_model(model_name, resolution, half=False, num_frames=50):
    """
    Benchmarks a YOLO model at a specific resolution and precision.
    Returns: (avg_latency_ms, fps, actual_half_used)
    """
    model = YOLO(model_name)

    # Check for synthetic frame override
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    # Try to open webcam, fallback to synthetic if not available or forced
    if force_synthetic:
        cap = None
        use_synthetic = True
    else:
        cap = cv2.VideoCapture(0)
        use_synthetic = not cap.isOpened()

    if use_synthetic:
        print(f"Using synthetic frames for {model_name} at {resolution}x{resolution}.")
        frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam. Falling back to synthetic.")
            use_synthetic = True
            frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)

    latencies = []

    # Warmup
    for _ in range(5):
        _ = model(frame, imgsz=resolution, half=half, verbose=False)

    # Benchmarking loop
    start_time = time.perf_counter()
    for _ in range(num_frames):
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret:
                break

        # Inference timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(frame, imgsz=resolution, half=half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    end_time = time.perf_counter()

    if cap is not None:
        cap.release()

    if not latencies:
        # Avoid division by zero
        return 0.0, 0.0, (half and torch.cuda.is_available())

    avg_latency = sum(latencies) / len(latencies)
    # FPS calculation based on total time for benchmarking loop
    fps = len(latencies) / (end_time - start_time)

    actual_half = half and torch.cuda.is_available()

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates a benchmarking result in the summary CSV.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '../results/tables/summary.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_entry = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(avg_fps, 2),
        'Average_Latency_ms': round(avg_latency, 2),
        'Observation': observation
    }

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Ensure required columns exist for matching
            if 'Model' not in df.columns:
                df['Model'] = 'yolov8n'
            if 'Precision' not in df.columns:
                df['Precision'] = 'FP32'

            # Match existing entry
            mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)

            if mask.any():
                df.loc[mask, ['Average_FPS', 'Average_Latency_ms', 'Observation']] = [
                    new_entry['Average_FPS'],
                    new_entry['Average_Latency_ms'],
                    new_entry['Observation']
                ]
            else:
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary.csv, creating new: {e}")
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])

    # Reorder columns to standard schema
    cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
    df = df[cols]

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
