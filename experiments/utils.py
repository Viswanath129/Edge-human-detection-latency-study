import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns (avg_latency_ms, fps, actual_half_used).
    """
    # Load model
    model = YOLO(model_path)

    # Check for half precision support
    actual_half = half
    if half:
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            print("Warning: CUDA not available, falling back to FP32.")
            actual_half = False

    # Frame source: physical webcam or synthetic fallback
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

        model(frame, imgsz=imgsz, verbose=False)

    # Benchmark loop
    latencies = []

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        # Measure inference-only time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark result in the central summary CSV.
    """
    file_path = 'results/tables/summary.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # Check for schema compatibility
            required_cols = ['Resolution', 'Model', 'Precision']
            if not all(col in df.columns for col in required_cols):
                print("Old schema detected, migrating...")
                # If it's the old schema from the initial state, it might miss Model and Precision
                if 'Model' not in df.columns:
                    df['Model'] = 'yolov8n' # assumption for old data
                if 'Precision' not in df.columns:
                    df['Precision'] = 'FP32'

            # Match existing entry
            mask = (df['Resolution'] == resolution) & \
                   (df['Model'] == model_name) & \
                   (df['Precision'] == precision)

            if mask.any():
                idx = df.index[mask][0]
                for col, val in new_data.items():
                    df.at[idx, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error updating summary: {e}. Creating new file.")
            pd.DataFrame([new_data]).to_csv(file_path, index=False)
    else:
        pd.DataFrame([new_data]).to_csv(file_path, index=False)
