import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, input_size, half=False, num_frames=100):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    model = YOLO(model_path)

    # Force CPU for consistency if not specified, but ultralytics handles it.
    # For half precision on CPU, it might be slow as noted in memory.

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    latencies = []

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        _ = model(frame, imgsz=input_size, half=half, verbose=False)

    start_bench = time.perf_counter()
    processed_frames = 0

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        _ = model(frame, imgsz=input_size, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        processed_frames += 1

    end_bench = time.perf_counter()

    if cap:
        cap.release()

    if processed_frames == 0:
        return 0.0, 0.0

    avg_latency = sum(latencies) / processed_frames
    avg_fps = processed_frames / (end_bench - start_bench)

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the summary.csv file.
    """
    summary_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = pd.DataFrame([{
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }])

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # Update if exists, else append
        mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
        else:
            df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")
