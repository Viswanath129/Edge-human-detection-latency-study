import cv2
import time
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model for latency and FPS.
    Returns (avg_latency, fps, actual_half)
    """
    model = YOLO(model_name)

    # Check if half precision is supported (CUDA required for meaningful results)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Force synthetic frames if requested or if no webcam
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found. Falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    print(f"Warming up {model_name} (imgsz={imgsz}, half={actual_half})...")
    for _ in range(5):
        frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    print(f"Benchmarking {model_name}...")

    start_time = time.perf_counter()
    frames_processed = 0

    try:
        for i in range(num_frames):
            if cap:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)
            frames_processed += 1

    except Exception as e:
        print(f"Error during benchmarking: {e}")
    finally:
        if cap:
            cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    end_time = time.perf_counter()
    avg_latency = sum(latencies) / len(latencies)
    fps = frames_processed / (end_time - start_time)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates the benchmarking results in results/tables/summary.csv
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../results/tables/summary.csv"))

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
            # Ensure columns exist
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Old schema detected. Overwriting summary.csv.")
                df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
        except Exception:
            df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])
    else:
        df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

    # Update or append
    mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
    if mask.any():
        for col, val in new_data.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
