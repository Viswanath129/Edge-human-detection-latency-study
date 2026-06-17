import time
import numpy as np
import cv2
import os
import pandas as pd
import torch

def benchmark_model(model, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model on either webcam or synthetic frames.
    Returns: (avg_latency_ms, fps, actual_half)
    """
    # Force synthetic frames in headless environments
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected. Falling back to synthetic frames.")
            cap = None

    # Warmup
    warmup_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(warmup_frame, imgsz=imgsz, half=half, verbose=False)

    latencies = []

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        # Synchronize for accurate timing if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        results = model(frame, imgsz=imgsz, half=half, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Check if FP16 was actually used (YOLOv8 might fallback if not supported)
    # For simplicity, we trust the 'half' parameter for now or check model properties

    return avg_latency, fps, half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmarking result in results/tables/summary.csv
    """
    summary_path = os.path.join(os.path.dirname(__file__), '../results/tables/summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        'Resolution': [resolution],
        'Model': [model_name],
        'Precision': [precision],
        'Average_FPS': [round(fps, 2)],
        'Average_Latency_ms': [round(latency, 2)],
        'Observation': [observation]
    }
    new_df = pd.DataFrame(new_data)

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Check if we need to update the schema of existing file
        if 'Model' not in df.columns or 'Precision' not in df.columns:
            print("Updating summary.csv schema...")
            # Default values for existing data
            if 'Model' not in df.columns:
                df['Model'] = 'yolov8n'
            if 'Precision' not in df.columns:
                df['Precision'] = 'FP32'

            # Reorder columns to match new schema
            cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
            df = df[cols]

        # Check if entry already exists (Resolution, Model, Precision)
        match = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if match.any():
            idx = df.index[match][0]
            df.loc[idx, 'Average_FPS'] = round(fps, 2)
            df.loc[idx, 'Average_Latency_ms'] = round(latency, 2)
            df.loc[idx, 'Observation'] = observation
        else:
            df = pd.concat([df, new_df], ignore_index=True)

        df.to_csv(summary_path, index=False)
    else:
        new_df.to_csv(summary_path, index=False)
