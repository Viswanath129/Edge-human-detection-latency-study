import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=20):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    # Load model
    model = YOLO(model_name)

    # Check for synthetic fallback
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected. Falling back to synthetic frames.")
            cap = None

    # Warmup phase (5 frames)
    warmup_frames = 5
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, half=half, verbose=False)

    latencies = []

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

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark entry in the summary CSV.
    """
    # Ensure directory exists
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "../results/tables/summary.csv")
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
        df = pd.read_csv(csv_path)
        # Check if entry exists to update
        mask = (df['Resolution'] == resolution) & \
               (df['Model'] == model_name) & \
               (df['Precision'] == precision)

        if mask.any():
            idx = df.index[mask][0]
            for key, val in new_data.items():
                df.at[idx, key] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
    print(f"Summary saved to {csv_path}")
