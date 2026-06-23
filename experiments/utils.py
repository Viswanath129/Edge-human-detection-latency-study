import os
import time
import cv2
import numpy as np
import pandas as pd
import torch

def benchmark_model(model, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks the given YOLO model.
    Returns (avg_latency_ms, fps, actual_half).
    """
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    use_webcam = not force_synthetic

    cap = None
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            use_webcam = False
            cap = None

    # Handle precision
    actual_half = False
    if half:
        if torch.cuda.is_available():
            try:
                model.to('cuda').half()
                actual_half = True
            except Exception:
                model.to('cpu').float()
        else:
            # Catching potential CPU FP16 issues
            try:
                model.half()
                actual_half = True
            except RuntimeError:
                model.float()
                actual_half = False

    latencies = []

    # Warmup
    for _ in range(5):
        if use_webcam and cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        _ = model(frame, imgsz=imgsz, verbose=False)

    # Main benchmark loop
    start_bench = time.perf_counter()
    frames_processed = 0

    for _ in range(num_frames):
        if use_webcam and cap:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        if actual_half and torch.cuda.is_available():
            torch.cuda.synchronize()

        _ = model(frame, imgsz=imgsz, verbose=False)

        if actual_half and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        frames_processed += 1

    end_bench = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = frames_processed / (end_bench - start_bench)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a benchmark result in results/tables/summary.csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "results", "tables", "summary.csv")
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
            # Schema migration check
            if "Model" not in df.columns or "Precision" not in df.columns:
                # Handle legacy schema
                df["Model"] = "yolov8n" if "Model" not in df.columns else df["Model"]
                df["Precision"] = "FP32" if "Precision" not in df.columns else df["Precision"]

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
        except Exception:
            df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_path, index=False)
