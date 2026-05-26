import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_frame_source():
    """Detects webcam or returns a synthetic frame generator if unavailable."""
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    if force_synthetic:
        print("FORCE_SYNTHETIC is true. Using synthetic frames.")
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected. Using synthetic frames.")
        return None
    return cap

def benchmark_model(model_name, resolution, precision="FP32", num_frames=30):
    """Benchmarks a YOLO model and returns average FPS and latency."""
    half = (precision == "FP16")
    model = YOLO(model_name)

    cap = get_frame_source()
    latencies = []

    # Warmup phase (5 frames)
    print(f"Starting 5-frame warmup for {model_name} at {resolution}...")
    warmup_frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(warmup_frame, imgsz=resolution, half=half, verbose=False)

    print(f"Benchmarking {model_name} (Precision: {precision}, Resolution: {resolution})...")

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (resolution, resolution))
        else:
            # Synthetic frame
            frame_resized = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)

        # Precise inference-only latency measurement
        t0 = time.perf_counter()
        _ = model(frame_resized, imgsz=resolution, half=half, verbose=False)
        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)

    if cap:
        cap.release()

    if not latencies:
        return 0.0, 0.0

    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0.0

    return avg_fps, avg_latency

def save_summary(resolution, model, precision, fps, latency, observation):
    """Saves benchmark results to the central summary CSV using absolute paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "../results/tables/summary.csv")

    new_data = {
        "Resolution": [f"{resolution}x{resolution}"],
        "Model": [model],
        "Precision": [precision],
        "Average_FPS": [round(fps, 2)],
        "Average_Latency_ms": [round(latency, 2)],
        "Observation": [observation]
    }
    new_df = pd.DataFrame(new_data)

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # Check if we need to add Model and Precision columns to old schema
        if "Model" not in df.columns:
            df["Model"] = "yolov8n" # Default for old entries
        if "Precision" not in df.columns:
            df["Precision"] = "FP32" # Default for old entries

        # Reorder columns to match schema
        df = df[["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]]

        # Remove existing entry for same config if exists
        mask = (df["Resolution"] == f"{resolution}x{resolution}") & \
               (df["Model"] == model) & \
               (df["Precision"] == precision)
        df = df[~mask]

        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")
