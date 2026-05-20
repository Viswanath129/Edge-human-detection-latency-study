import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_frame_source(force_synthetic=False):
    """
    Detects frame source: webcam or synthetic frames.
    """
    if force_synthetic:
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected. Falling back to synthetic frames.")
        return None
    return cap

def run_benchmark(model_path, img_size=640, half=False, num_frames=20, warmup_frames=5, experiment_name="test"):
    """
    Runs a benchmark for a given model and configuration.
    """
    # Load model
    model = YOLO(model_path)

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = get_frame_source(force_synthetic)

    latencies = []

    print(f"Starting benchmark: {experiment_name} (Size={img_size}, Half={half})")

    # Warmup phase
    for i in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        _ = model(frame, imgsz=img_size, half=half, verbose=False)

    # Benchmark loop
    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        t0 = time.time()
        _ = model(frame, imgsz=img_size, half=half, verbose=False)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000)  # ms

    if cap:
        cap.release()

    if not latencies:
        print(f"Error: No latencies recorded for {experiment_name}. Skipping...")
        return {
            "experiment": experiment_name,
            "resolution": img_size,
            "avg_latency_ms": None,
            "fps": None
        }

    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency

    print(f"Finished: Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

    # Save raw results
    os.makedirs("results/tables", exist_ok=True)
    df_raw = pd.DataFrame({"latency_ms": latencies})
    df_raw.to_csv(f"results/tables/{experiment_name}_raw.csv", index=False)

    return {
        "experiment": experiment_name,
        "resolution": img_size,
        "avg_latency_ms": avg_latency,
        "fps": fps
    }
