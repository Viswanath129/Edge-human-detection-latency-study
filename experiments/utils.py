import cv2
import time
import numpy as np
import os
import pandas as pd
import torch

def benchmark_model(model, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks the model for a given number of frames.
    Falls back to synthetic frames if no webcam is available.
    """
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            cap = None
        else:
            # Check if we can actually read a frame
            ret, _ = cap.read()
            if not ret:
                cap.release()
                cap = None

    if cap is None:
        print("Using synthetic frames for benchmarking.")

    latencies = []

    # Check if half precision is supported (CUDA only for YOLOv8 typically)
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print("FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Warmup
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency
    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, avg_fps, avg_latency, observation=""):
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables/summary.csv"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(avg_fps, 2),
        "Average_Latency_ms": round(avg_latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Ensure necessary columns exist for old files
            if "Model" not in df.columns:
                df["Model"] = "yolov8n"
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"

            # Update if exists, else append
            mask = (df['Resolution'] == resolution) & (df['Model'] == model_name) & (df['Precision'] == precision)
            if mask.any():
                for col, val in new_data.items():
                    df.loc[mask, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
