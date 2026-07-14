import cv2
import time
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50, force_synthetic=False):
    """
    Benchmarks a YOLO model and returns (avg_latency, fps, actual_half).
    """
    model = YOLO(model_path)

    # Check for CUDA and half precision support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual_half = half and device == 'cuda'

    if actual_half:
        model.to(device).half()
    else:
        model.to(device)

    # Setup frame source
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)

    # Benchmark loop
    latencies = []
    start_time = time.perf_counter()

    # Pre-generate synthetic frame if needed to minimize overhead
    synthetic_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) if not cap else None

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = synthetic_frame

        t0 = time.perf_counter()
        if device == 'cuda':
            torch.cuda.synchronize()

        _ = model(frame, imgsz=imgsz, verbose=False, half=actual_half)

        if device == 'cuda':
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
    Saves or updates the benchmark result in results/tables/summary.csv.
    """
    # Normalize model name
    if model_name.lower().startswith('yolov8'):
        model_name = 'YOLOv8' + model_name[6:].lower()

    # Path setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, 'results', 'tables', 'summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Schema migration/check
        if 'Model' not in df.columns:
            df['Model'] = 'YOLOv8n'
        if 'Precision' not in df.columns:
            df['Precision'] = 'FP32'

        # Match by Resolution, Model, and Precision
        match = (df['Resolution'] == resolution) & \
                (df['Model'] == model_name) & \
                (df['Precision'] == precision)

        if match.any():
            idx = df.index[match][0]
            for key, val in new_entry.items():
                df.at[idx, key] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    # Ensure columns are in the correct order
    cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
    df = df[cols]

    df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")
