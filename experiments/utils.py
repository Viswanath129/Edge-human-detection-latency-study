import cv2
import time
import pandas as pd
import numpy as np
import torch
import os
from ultralytics import YOLO

def benchmark_model(model_path, imgsz=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model with the given settings.
    Returns: (avg_latency_ms, fps, actual_half)
    """
    # Load model (automatically downloads weights locally if not present)
    model = YOLO(model_path)

    # Check CUDA and FP16 availability
    cuda_available = torch.cuda.is_available()
    if half and not cuda_available:
        print("Warning: FP16 requested but CUDA not available. Results may be non-representative on CPU.")

    # Frame source detection: prioritize physical camera, fallback to synthetic
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None
    use_webcam = False

    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            use_webcam = True
        else:
            if cap is not None:
                cap.release()
            print("Webcam not detected or failed to open. Falling back to synthetic frames.")

    # 5-frame warmup using the same precision settings to stabilize inference
    actual_half = half
    warmup_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        try:
            model(warmup_frame, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError as e:
            if actual_half:
                print(f"FP16 not supported or errored, falling back to FP32: {e}")
                actual_half = False
                model(warmup_frame, imgsz=imgsz, half=actual_half, verbose=False)
            else:
                raise e

    # Hardened inference loop with timed execution (inference-only FPS calculation)
    # Pre-generate synthetic frames outside of the timed inference loop to minimize acquisition overhead
    synthetic_frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(num_frames)]
    latencies = []

    for idx in range(num_frames):
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam during benchmark, falling back to pre-generated synthetic frame.")
                frame = synthetic_frames[idx]
            else:
                frame = cv2.resize(frame, (imgsz, imgsz))
        else:
            frame = synthetic_frames[idx]

        # Synchronize GPU if CUDA is active
        if cuda_available:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # Run model inference
        model(frame, imgsz=imgsz, half=actual_half, verbose=False)

        if cuda_available:
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    if cap is not None:
        cap.release()

    if not latencies:
        return 0.0, 0.0, actual_half

    # Calculate and return inference-only FPS and average latency
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, fps, latency, observation):
    """
    Saves and updates benchmark findings in results/tables/summary.csv.
    Standardized schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation
    Normalizes model names to 'YOLOv8' prefix with lowercase suffix (e.g. yolov8n -> YOLOv8n)
    """
    # Normalize model name
    normalized_model_name = model_name
    if model_name.lower().startswith("yolov8"):
        suffix = model_name[6:].lower()
        normalized_model_name = f"YOLOv8{suffix}"

    # Use absolute paths derived from this file location to ensure correct target file resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "../results/tables/summary.csv"))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    new_entry = pd.DataFrame([{
        "Resolution": resolution,
        "Model": normalized_model_name,
        "Precision": precision,
        "Average_FPS": round(fps, 2),
        "Average_Latency_ms": round(latency, 2),
        "Observation": observation
    }])

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)

            # Detect missing 'Model' or 'Precision' columns to handle schema migration safely
            if "Model" not in df.columns or "Precision" not in df.columns:
                print("Missing 'Model' or 'Precision' column in existing summary.csv. Recreating table.")
                df = new_entry
            else:
                # Standardize matching to update existing entry instead of duplicate appending
                mask = (df['Resolution'] == resolution) & (df['Model'] == normalized_model_name) & (df['Precision'] == precision)
                if mask.any():
                    df.loc[mask, ["Average_FPS", "Average_Latency_ms", "Observation"]] = [round(fps, 2), round(latency, 2), observation]
                else:
                    df = pd.concat([df, new_entry], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing summary.csv, overwriting: {e}")
            df = new_entry
    else:
        df = new_entry

    df.to_csv(summary_path, index=False)
    print(f"Successfully saved entry to {summary_path}")
