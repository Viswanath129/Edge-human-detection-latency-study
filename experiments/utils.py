import cv2
import time
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=20, warmup_frames=5):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns: (avg_fps, avg_latency_ms)
    """
    model = YOLO(model_name)

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None

    latencies = []

    # Warmup phase
    for _ in range(warmup_frames):
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

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    # Inference-only FPS calculation
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return fps, avg_latency

def save_summary(resolution, model_variant, precision, fps, latency, observation):
    """
    Saves or appends results to results/tables/summary.csv
    """
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    columns = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    new_data = pd.DataFrame([{
        "Resolution": resolution,
        "Model": model_variant,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }])

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Ensure all columns exist
            for col in columns:
                if col not in df.columns:
                    df[col] = "N/A"
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df = pd.DataFrame(columns=columns)

        # Check if the exact configuration already exists and update it, or append
        mask = (df['Resolution'] == str(resolution)) & (df['Model'] == model_variant) & (df['Precision'] == precision)
        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
        else:
            df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
