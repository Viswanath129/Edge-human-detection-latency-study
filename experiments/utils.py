import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name="yolov8n.pt", imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency_ms, fps, actual_half).
    """
    # Load model
    model = YOLO(model_name)

    # Check if half precision is possible
    actual_half = half
    if half and not torch.cuda.is_available():
        print(f"Warning: FP16 requested but CUDA not available. Falling back to FP32.")
        actual_half = False

    # Try to use webcam unless forced synthetic
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected. Falling back to synthetic frames.")
            cap = None

    # Pre-generate a synthetic frame if needed to avoid overhead in the loop
    synthetic_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up {model_name} at {imgsz}x{imgsz} (half={actual_half})...")
    for _ in range(5):
        model(synthetic_frame, imgsz=imgsz, half=actual_half, verbose=False)

    latencies = []
    print(f"Benchmarking...")

    start_total = time.perf_counter()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
            frame_input = cv2.resize(frame, (imgsz, imgsz))
        else:
            frame_input = synthetic_frame

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model(frame_input, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    end_total = time.perf_counter()

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = len(latencies) / (end_total - start_total)

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates a row in results/tables/summary.csv.
    """
    # Normalize model name
    if model_name.startswith("yolov8"):
        model_display = "YOLOv8" + model_name[6].lower() if len(model_name) > 6 else "YOLOv8n"
    else:
        model_display = model_name

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, "results", "tables", "summary.csv")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_data = {
        "Resolution": resolution,
        "Model": model_display,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Handle legacy schema migration if necessary
        if "Model" not in df.columns or "Precision" not in df.columns:
            print("Migrating legacy summary.csv to new schema...")
            if "Model" not in df.columns:
                df["Model"] = "YOLOv8n"
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"

        # Check for existing entry to update
        mask = (df["Resolution"] == resolution) & (df["Model"] == model_display) & (df["Precision"] == precision)
        if mask.any():
            for col, val in new_data.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
