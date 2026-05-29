import os
import time
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=20, force_synthetic=False):
    """
    Benchmarks a YOLO model for latency and FPS.
    Supports physical webcam or synthetic frames.
    """
    # Load model
    model = YOLO(model_path)

    # Check if webcam is available
    cap = None
    if not force_synthetic and os.environ.get('FORCE_SYNTHETIC', 'false').lower() != 'true':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    latencies = []

    # Warmup (5 frames)
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=half, verbose=False)

    # Benchmark loop
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        start = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv.
    """
    summary_path = os.path.join(os.path.dirname(__file__), '../results/tables/summary.csv')
    summary_path = os.path.abspath(summary_path)

    df_new = pd.DataFrame([{
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }])

    if os.path.exists(summary_path):
        df_old = pd.read_csv(summary_path)
        # Drop existing entry if it matches Resolution, Model, Precision to update it
        df_old = df_old[~((df_old['Resolution'] == str(resolution)) &
                          (df_old['Model'] == model_name) &
                          (df_old['Precision'] == precision))]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
