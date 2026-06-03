import cv2
import time
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=20):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    model = YOLO(model_path)

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=half, verbose=False)

    # Inference loop
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
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0.0

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark results in results/tables/summary.csv.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
    csv_path = os.path.abspath(csv_path)

    new_entry = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check for existing entry to update
        match = (df['Resolution'] == resolution) & \
                (df['Model'] == model_name) & \
                (df['Precision'] == precision)

        if match.any():
            for key, value in new_entry.items():
                df.loc[match, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
