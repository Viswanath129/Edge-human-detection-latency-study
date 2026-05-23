import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def run_benchmark(model_name, resolution, half=False, num_frames=100, observation=""):
    """
    Runs a benchmark for a specific model configuration.
    """
    print(f"Benchmarking {model_name} at {resolution}x{resolution} (half={half})...")

    # Load model
    model = YOLO(model_name)

    # Absolute paths for saving results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables_dir = os.path.join(base_dir, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    summary_path = os.path.join(tables_dir, "summary.csv")

    # Warmup phase (5 frames)
    warmup_frames = 5
    for _ in range(warmup_frames):
        dummy_frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        _ = model(dummy_frame, half=half, verbose=False)

    latencies = []

    # Frame source detection
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    if force_synthetic:
        source_name = "synthetic"
        for _ in range(num_frames):
            frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            t0 = time.time()
            _ = model(frame, half=half, verbose=False)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, falling back to synthetic frames.")
            source_name = "synthetic_fallback"
            for _ in range(num_frames):
                frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                t0 = time.time()
                _ = model(frame, half=half, verbose=False)
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)
        else:
            source_name = "webcam"
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (resolution, resolution))
                t0 = time.time()
                _ = model(frame_resized, half=half, verbose=False)
                t1 = time.time()
                latencies.append((t1 - t0) * 1000)
            cap.release()

    # Metrics calculation
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    precision = "FP16" if half else "FP32"
    model_short = model_name.split('.')[0]

    result = {
        "Resolution": f"{resolution}x{resolution}",
        "Model": model_short,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(avg_latency, 2),
        "Observation": observation
    }

    # Dual-mode saving
    # 1. Append to summary.csv
    df_new = pd.DataFrame([result])
    if not os.path.exists(summary_path):
        df_new.to_csv(summary_path, index=False)
    else:
        # Check if already exists to avoid duplicates (optional but good)
        df_existing = pd.read_csv(summary_path)
        # Simple append for now
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(summary_path, index=False)

    # 2. Save raw latencies
    log_filename = f"{model_short}_{resolution}_{precision}_raw.csv"
    log_path = os.path.join(tables_dir, log_filename)
    pd.DataFrame({"latency_ms": latencies}).to_csv(log_path, index=False)

    print(f"Benchmark complete. Results saved to {summary_path}")
    return result
