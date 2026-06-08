import cv2
import time
import torch
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if half and device == 'cpu':
        print("Warning: FP16 is not well-supported on CPU. Falling back to FP32 or expecting slow performance.")

    try:
        model = YOLO(model_path)
        if half and device == 'cuda':
            model.to(device).half()
        else:
            model.to(device)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return 0.0, 0.0

    # Frame source
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Falling back to synthetic frames.")
            cap = None

    # Warmup
    print(f"Warming up {model_path} at {imgsz}x{imgsz} (half={half})...")
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model(frame, imgsz=imgsz, verbose=False)

    # Benchmark loop
    print(f"Benchmarking {model_path} for {num_frames} frames...")
    latencies = []

    total_start = time.perf_counter()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    total_end = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = len(latencies) / (total_end - total_start)

    return avg_fps, avg_latency

def save_summary(resolution, model_name, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates the benchmark results in results/tables/summary.csv.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(avg_fps, 2),
        "Average_Latency_ms": round(avg_latency, 2),
        "Observation": observation
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Check if the entry already exists
        match = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
        if match.any():
            for key, value in new_data.items():
                df.loc[match, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
