import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model, input_size=640, num_frames=20, warmup_frames=5, **inference_kwargs):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    if force_synthetic:
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        _ = model(frame, imgsz=input_size, verbose=False, **inference_kwargs)

    # Benchmark loop
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=input_size, verbose=False, **inference_kwargs)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    # FPS based on inference-only latency to be technically precise
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark results in results/tables/summary.csv.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "../results/tables/summary.csv")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Ensure all columns exist
            for col in new_data.keys():
                if col not in df.columns:
                    df[col] = None
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame(columns=new_data.keys())
    else:
        df = pd.DataFrame(columns=new_data.keys())

    # Check if entry exists by matching Resolution, Model, and Precision
    match = (df["Resolution"] == resolution) & \
            (df["Model"] == model_name) & \
            (df["Precision"] == precision)

    if match.any():
        idx = df.index[match][0]
        for col, val in new_data.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
