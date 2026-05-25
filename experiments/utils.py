import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def run_benchmark(model_path, imgsz=640, half=False, num_frames=50):
    """
    Runs a benchmark for a given YOLO model.
    """
    print(f"Benchmarking {model_path} at {imgsz}px (half={half})...")

    # Load model (automatically downloads if not present)
    model = YOLO(model_path)

    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, falling back to synthetic frames.")
            cap = None

    latencies = []

    # Use a synthetic frame if no webcam
    if cap is None:
        frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    print("Warmup phase...")
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (imgsz, imgsz))
        model(frame, imgsz=imgsz, half=half, verbose=False)

    print("Inference phase...")
    start_bench = time.perf_counter()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (imgsz, imgsz))

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    end_bench = time.perf_counter()

    if cap:
        cap.release()

    avg_latency = np.mean(latencies)
    fps = len(latencies) / (end_bench - start_bench)

    print(f"Results: Avg Latency = {avg_latency:.2f}ms, FPS = {fps:.2f}")

    return {
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "raw_latencies": latencies
    }

def save_summary(results_list, output_path="results/tables/summary.csv"):
    """
    Updates the central summary CSV with new results.
    Standardizes schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    full_output_path = os.path.join(root_dir, output_path)

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

    new_df = pd.DataFrame(results_list)

    if os.path.exists(full_output_path):
        old_df = pd.read_csv(full_output_path)
        combined_df = pd.concat([old_df, new_df]).drop_duplicates(
            subset=["Resolution", "Model", "Precision"], keep="last"
        )
    else:
        combined_df = new_df

    combined_df.to_csv(full_output_path, index=False)
    print(f"Summary updated at {full_output_path}")
