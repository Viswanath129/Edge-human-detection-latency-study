import cv2
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

def run_benchmark(model_path, resolution=640, half=False, observation=""):
    """
    Runs a benchmark for a given YOLO model and resolution.

    Args:
        model_path (str): Path to the YOLO model file (.pt).
        resolution (int): Input resolution for the model.
        half (bool): Use half precision (FP16).
        observation (str): Observation note for the benchmark.
    """
    model_name = os.path.basename(model_path)
    precision = "FP16" if half else "FP32"

    print(f"Benchmarking {model_name} at {resolution}x{resolution} ({precision})...")

    # Load model
    model = YOLO(model_path)

    # Check for FORCE_SYNTHETIC environment variable
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not detected. Falling back to synthetic frames.")
            cap = None

    # Warmup phase (5 frames)
    warmup_frames = 5
    dummy_frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    for _ in range(warmup_frames):
        model(dummy_frame, conf=0.4, iou=0.5, half=half, verbose=False)

    latencies = []
    num_frames = 50

    for i in range(num_frames):
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                frame = dummy_frame
            else:
                frame = cv2.resize(frame, (resolution, resolution))
        else:
            frame = dummy_frame

        t0 = time.perf_counter()
        model(frame, conf=0.4, iou=0.5, half=half, verbose=False)
        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)

    if cap is not None:
        cap.release()

    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "tables"))
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.csv")

    new_data = {
        "Resolution": f"{resolution}x{resolution}",
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(avg_latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # Check if entry already exists and update, or append
        mask = (df['Resolution'] == new_data['Resolution']) & \
               (df['Model'] == new_data['Model']) & \
               (df['Precision'] == new_data['Precision'])

        if mask.any():
            df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = \
                [new_data["Average_FPS"], new_data["Average_Latency_ms"], new_data["Observation"]]
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)

    # Save raw latencies
    raw_filename = f"raw_{model_name}_{resolution}_{precision}.csv"
    raw_path = os.path.join(results_dir, raw_filename)
    pd.DataFrame({"latency_ms": latencies}).to_csv(raw_path, index=False)

    print(f"Results saved to {summary_path} and {raw_path}")
