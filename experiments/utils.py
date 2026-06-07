import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, precision='FP32', num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    # Load model
    model = YOLO(model_path)

    # Handle precision with safety checks
    if precision == 'FP16':
        if torch.cuda.is_available():
            model.to('cuda').half()
        else:
            # On CPU, half precision is often not supported for many ops
            # We attempt it but catch errors, or just note it's not optimized
            try:
                model.to('cpu').half()
            except RuntimeError as e:
                print(f"Warning: FP16 not fully supported on this CPU: {e}")
                # Fallback or continue if only some ops fail

    # Check for synthetic frame enforcement
    force_synthetic = os.environ.get('FORCE_SYNTHETIC', 'false').lower() == 'true'

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    # Warmup phase (5 frames)
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        try:
            _ = model(frame, imgsz=imgsz, verbose=False)
        except Exception:
            pass

    latencies = []

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        try:
            _ = model(frame, imgsz=imgsz, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
        except RuntimeError as e:
            print(f"Inference error: {e}")
            break

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return fps, avg_latency

def save_summary(resolution, model, precision, fps, latency, observation):
    """
    Saves or updates the benchmarking results in results/tables/summary.csv
    """
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/tables/summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = {
        'Resolution': resolution,
        'Model': model,
        'Precision': precision,
        'Average_FPS': round(fps, 2),
        'Average_Latency_ms': round(latency, 2),
        'Observation': observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Filter out legacy entries with missing 'Model' or 'Precision' if they exist
            if 'Model' in df.columns:
                df = df[df['Model'].notna()]
        except Exception:
            df = pd.DataFrame(columns=new_entry.keys())

        # Ensure all columns exist
        for col in new_entry.keys():
            if col not in df.columns:
                df[col] = None

        # Check for existing entry to update
        match = (df['Resolution'] == resolution) & (df['Model'] == model) & (df['Precision'] == precision)
        if match.any():
            idx = df.index[match][0]
            for col, val in new_entry.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(summary_path, index=False)
    print(f"Summary updated for {model} at {resolution} ({precision})")
