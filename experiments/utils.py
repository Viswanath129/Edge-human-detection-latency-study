import cv2
import time
import os
import numpy as np
import pandas as pd
import torch

def benchmark_model(model, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model and returns (avg_latency_ms, fps, actual_half).
    """
    # Check for CUDA and FP16 support
    actual_half = half and torch.cuda.is_available()
    if half and not torch.cuda.is_available():
        print("Warning: FP16 requested but CUDA is not available. Falling back to FP32.")

    # Setup frame source
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found, using synthetic frames.")
            cap = None

    # Pre-generate synthetic frames if needed to avoid overhead in the loop
    synthetic_frames = []
    if not cap:
        for _ in range(5):
            synthetic_frames.append(np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8))

    def get_frame(i):
        if cap:
            ret, frame = cap.read()
            if ret:
                return frame
        return synthetic_frames[i % len(synthetic_frames)]

    # Warmup
    for i in range(5):
        frame = get_frame(i)
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    t_start = time.perf_counter()

    for i in range(num_frames):
        frame = get_frame(i)

        t0 = time.perf_counter()
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    t_end = time.perf_counter()

    if cap:
        cap.release()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    # FPS should be based on inference-only time to be representative of model capability
    # or based on total time if including I/O. We'll use inference-only FPS for technical precision.
    inference_only_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, inference_only_fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves or updates the benchmark results in results/tables/summary.csv.
    """
    # Use absolute path relative to this file to avoid issues with different CWDs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    summary_path = os.path.join(project_root, "results", "tables", "summary.csv")

    new_data = {
        "Resolution": resolution,
        "Model": model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Check if columns exist (migration)
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Updating legacy summary.csv schema...")
                df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

            # Find if entry exists
            mask = (df["Resolution"] == resolution) & (df["Model"] == model_name) & (df["Precision"] == precision)
            if mask.any():
                df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception as e:
            print(f"Error reading summary.csv: {e}. Creating new.")
            df = pd.DataFrame([new_data])
    else:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        df = pd.DataFrame([new_data])

    df.to_csv(summary_path, index=False)
    print(f"Summary updated at {summary_path}")
