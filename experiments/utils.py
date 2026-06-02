import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=20):
    """
    Benchmarks a YOLO model for latency and FPS.
    Hardened against division-by-zero and supports synthetic fallback.
    """
    try:
        # Load model
        model = YOLO(model_name)
        if half:
            # FP16 is only effective on GPU, but we follow the request
            if torch.cuda.is_available():
                model.to('cuda')
                model.half()
            else:
                # Fallback to CPU, FP16 might be slower
                model.to('cpu')
                # Note: CPU FP16 isn't well supported in some torch versions,
                # but we'll try if it was requested.
                try:
                    model.model.half()
                except:
                    pass
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        model = YOLO(model_name)

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    latencies = []

    # Warmup phase (5 frames)
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        _ = model(frame, imgsz=imgsz, verbose=False)

    # Inference loop
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    # Inference-only FPS: strictly based on latency
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Updates the central results/tables/summary.csv with standardized schema.
    Matches existing entries by Resolution, Model, and Precision.
    """
    # Use absolute path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, '../results/tables/summary.csv'))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
        'Resolution': resolution,
        'Model': model_name,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    columns = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)

            # Schema check and migration
            if not all(col in df.columns for col in columns):
                if 'Resolution' in df.columns and 'Model' not in df.columns:
                    df['Model'] = 'yolov8n'
                    df['Precision'] = 'FP32'
                    df = df[columns]

            # Identify existing entry
            match = (df['Resolution'] == resolution) & \
                    (df['Model'] == model_name) & \
                    (df['Precision'] == precision)

            if match.any():
                df.loc[match, ['Average_FPS', 'Average_Latency_ms', 'Observation']] = \
                    [new_entry['Average_FPS'], new_entry['Average_Latency_ms'], new_entry['Observation']]
            else:
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary, creating new: {e}")
            df = pd.DataFrame([new_entry])
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(summary_path, index=False)
    print(f"Updated summary at {summary_path}")
