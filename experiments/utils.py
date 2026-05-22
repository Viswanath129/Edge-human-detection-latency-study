import cv2
import time
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO

def get_frame_source(force_synthetic=False):
    if force_synthetic:
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    return cap

def run_benchmark(model_path, img_size=640, half=False, num_frames=20, warmup_frames=2):
    model = YOLO(model_path)

    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = get_frame_source(force_synthetic)

    latencies = []

    # Warmup
    for _ in range(warmup_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        model(frame, imgsz=img_size, half=half, verbose=False)

    start_time = time.time()
    for _ in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        t0 = time.time()
        model(frame, imgsz=img_size, half=half, verbose=False)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)

    end_time = time.time()
    if cap:
        cap.release()

    avg_latency = np.mean(latencies)
    fps = len(latencies) / (end_time - start_time)

    return fps, avg_latency, latencies

def save_results(resolution, model_name, precision, fps, avg_latency, latencies, observation=""):
    # Save raw latencies
    raw_df = pd.DataFrame({"latency_ms": latencies})
    os.makedirs("results/tables", exist_ok=True)
    raw_filename = f"results/tables/{model_name}_{resolution}_{precision}_raw.csv"
    raw_df.to_csv(raw_filename, index=False)

    # Update summary
    summary_path = "results/tables/summary.csv"
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
        # Check if we need to add columns
        for col in new_data.keys():
            if col not in df.columns:
                df[col] = None

        # Check if entry exists to update or append
        mask = (df['Resolution'] == new_data['Resolution']) & \
               (df['Model'] == new_data['Model']) & \
               (df['Precision'] == new_data['Precision'])

        if mask.any():
            for col, val in new_data.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    # Clean up old manual entries if they exist
    df = df[df['Model'].notna()]

    df.to_csv(summary_path, index=False)
