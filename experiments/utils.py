import cv2
import time
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def run_benchmark(model_path, imgsz=640, half=False, num_frames=30, experiment_name="test"):
    """
    Runs a benchmark for a given model and configuration.
    """
    print(f"Running benchmark: {experiment_name} (imgsz={imgsz}, half={half})")

    # Load model
    model = YOLO(model_path)
    if half and not torch.cuda.is_available():
        print("WARNING: FP16 (half=True) is not optimized for CPU. Performance will be poor.")

    if half:
        model.to('cuda' if torch.cuda.is_available() else 'cpu') # Ensure device is set
        # model.half() is called during inference via the 'half' parameter in ultralytics

    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, falling back to synthetic frames.")
            cap = None

    latencies = []

    # Warmup
    print("Warming up...")
    for _ in range(5):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        model(frame, imgsz=imgsz, half=half, verbose=False)

    print(f"Benchmarking {num_frames} frames...")
    start_time = time.time()
    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        t0 = time.time()
        model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000) # ms

    end_time = time.time()

    if cap:
        cap.release()

    if not latencies:
        print("No frames processed.")
        return

    avg_latency = np.mean(latencies)
    # Inference-only FPS
    fps = 1000 / avg_latency

    print(f"Results for {experiment_name}:")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  Inference FPS: {fps:.2f}")

    # Save results
    os.makedirs("results/tables", exist_ok=True)

    # Save raw latencies
    raw_df = pd.DataFrame({"latency_ms": latencies})
    raw_df.to_csv(f"results/tables/{experiment_name}_raw.csv", index=False)

    # Save summary
    summary_path = "results/tables/summary.csv"
    summary_data = {
        "experiment": experiment_name,
        "imgsz": imgsz,
        "half": half,
        "avg_latency_ms": avg_latency,
        "fps": fps
    }
    summary_df = pd.DataFrame([summary_data])

    if os.path.exists(summary_path):
        try:
            existing_summary = pd.read_csv(summary_path)
            # Update or append
            if 'experiment' in existing_summary.columns:
                if experiment_name in existing_summary['experiment'].values:
                    existing_summary = existing_summary[existing_summary['experiment'] != experiment_name]
                summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary file: {e}. Starting fresh.")

    summary_df.to_csv(summary_path, index=False)
    return summary_data
