import cv2
import time
import pandas as pd
import numpy as np
import torch
import os
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=20):
    # Load model
    model = YOLO(model_path)

    # FP16 check: only use if CUDA is available, otherwise it's often slower on CPU
    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA not available. Results may be non-representative on CPU.")

    # Warmup
    actual_half = half
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        try:
            model(dummy_frame, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError:
            if actual_half:
                print("FP16 not supported on this device, falling back to FP32")
                actual_half = False
                model(dummy_frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []

    # Run benchmark
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    summary_path = "results/tables/summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = pd.DataFrame([{
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }])

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Basic validation of existing columns
            if "Model" not in df.columns or "Precision" not in df.columns:
                # Handle legacy schema
                df = new_entry
            else:
                mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
                if mask.any():
                    df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
                else:
                    df = pd.concat([df, new_entry], ignore_index=True)
        except Exception:
            df = new_entry
    else:
        df = new_entry

    df.to_csv(summary_path, index=False)
