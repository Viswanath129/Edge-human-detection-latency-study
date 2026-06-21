import cv2
import time
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def benchmark_model(model_name="yolov8n.pt", imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and FPS.
    Returns (avg_latency, fps, actual_half)
    """
    model = YOLO(model_name)

    # Check if half precision is supported and requested
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        half = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Warmup
    print(f"Warming up {model_name} at {imgsz}...")
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy_frame, imgsz=imgsz, half=half, verbose=False)

    # Try to open webcam, fallback to synthetic if fails
    cap = cv2.VideoCapture(0)
    use_synthetic = not cap.isOpened()

    if use_synthetic:
        print("Webcam not detected. Using synthetic frames for benchmarking.")
    else:
        print("Using webcam for benchmarking.")

    latencies = []

    print(f"Starting benchmark for {num_frames} frames...")
    start_time = time.perf_counter()

    for _ in range(num_frames):
        if use_synthetic:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (imgsz, imgsz))

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000) # ms

    end_time = time.perf_counter()

    if not use_synthetic:
        cap.release()

    if not latencies:
        return 0.0, 0.0, half

    avg_latency = sum(latencies) / len(latencies)
    total_time = end_time - start_time
    fps = len(latencies) / total_time

    return avg_latency, fps, half

def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates the benchmarking results in results/tables/summary.csv
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "tables", "summary.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    new_data = {
        "Resolution": f"{resolution}x{resolution}",
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure schema matches
        if "Model" not in df.columns or "Precision" not in df.columns:
            # Legacy format, just overwrite or expand
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

        # Check if entry already exists to update it
        match = (df["Resolution"] == new_data["Resolution"]) & \
                (df["Model"] == new_data["Model"]) & \
                (df["Precision"] == new_data["Precision"])

        if match.any():
            idx = df.index[match][0]
            for col, val in new_data.items():
                df.at[idx, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
