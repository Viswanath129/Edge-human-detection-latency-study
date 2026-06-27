import cv2
import time
import os
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency_ms, fps, actual_half_used)
    """
    model = YOLO(model_name)

    # Check if half precision is possible (only on CUDA)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found or FORCE_SYNTHETIC=true. Using synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    start_time = time.perf_counter()
    processed_frames = 0

    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        processed_frames += 1

    end_time = time.perf_counter()

    if cap:
        cap.release()

    if processed_frames == 0:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / processed_frames
    fps = processed_frames / (end_time - start_time)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "../results/tables/summary.csv")

    # Ensure directory exists
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
            # Schema migration if needed
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Updating legacy summary.csv schema...")
                # Best effort to fill legacy data
                if "Model" not in df.columns: df["Model"] = "yolov8n"
                if "Precision" not in df.columns: df["Precision"] = "FP32"

            # Check for existing entry to update
            match = (df["Resolution"] == resolution) & \
                    (df["Model"] == model_name) & \
                    (df["Precision"] == precision)

            if match.any():
                idx = df.index[match][0]
                for col, val in new_data.items():
                    df.at[idx, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary.csv: {e}. Creating new one.")
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
