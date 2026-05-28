import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_path, input_size=640, half=False, num_frames=20, save_path=None):
    """
    Benchmarks a YOLO model for latency and FPS.
    """
    model = YOLO(model_path)

    # Check for webcam or force synthetic
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected, falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    print(f"Warming up {model_path} at {input_size}x{input_size}...")
    for _ in range(5):
        dummy_frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        model(dummy_frame, imgsz=input_size, half=half, verbose=False)

    print(f"Benchmarking {model_path} (half={half})...")
    start_time = time.time()

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (input_size, input_size))
        else:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model(frame, imgsz=input_size, half=half, verbose=False)
        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)

    end_time = time.time()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (end_time - start_time)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame({"latency_ms": latencies})
        df.to_csv(save_path, index=False)

    return avg_latency, fps

def save_summary(data, filename="results/tables/summary.csv"):
    """
    Saves or appends benchmark results to a summary CSV.
    Data should be a list of dicts: [
        {"Resolution": "640x640", "Model": "yolov8n", "Precision": "FP32", "Average_FPS": 7.6, "Average_Latency_ms": 110.0, "Observation": "..."},
        ...
    ]
    """
    # Ensure directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    full_path = os.path.join(root_dir, filename)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(full_path, index=False)
    print(f"Summary saved to {full_path}")
