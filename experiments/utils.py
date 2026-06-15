import cv2
import time
import torch
import numpy as np
import pandas as pd
import os

def benchmark_model(model, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency_ms, fps, actual_half)
    """
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
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
            if not ret:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, verbose=False)

    latencies = []
    actual_half = half

    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA is not available. Falling back to FP32.")
        actual_half = False

    try:
        for _ in range(num_frames):
            if cap:
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            else:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            model(frame, imgsz=imgsz, half=actual_half, verbose=False)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)
    except RuntimeError as e:
        print(f"RuntimeError during benchmarking: {e}")
        if half:
            print("Likely FP16 not supported on this hardware.")
        return 0.0, 0.0, False
    finally:
        if cap:
            cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation, csv_path="results/tables/summary.csv"):
    """
    Saves or updates benchmark results in a central summary CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Check for schema migration
            if "Model" not in df.columns or "Precision" not in df.columns:
                raise ValueError("Old schema detected")

            # Update if match found
            mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
            if mask.any():
                df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except (ValueError, pd.errors.EmptyDataError):
            # Overwrite or start fresh on error/migration
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
