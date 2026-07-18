import os
import time
import cv2
import torch
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name_or_path, imgsz, half=False, num_frames=50):
    """
    Benchmarks a YOLO model's inference-only latency and FPS.
    Supports physical webcam frame capture or falls back to synthetic frames if unavailable
    or if FORCE_SYNTHETIC=true is specified.

    Args:
        model_name_or_path (str): Path or identifier of the model (e.g. 'yolov8n.pt').
        imgsz (int): Input resolution for inference.
        half (bool): Whether to use FP16 half precision.
        num_frames (int): Number of inference frames.

    Returns:
        tuple: (average_latency_ms, inference_only_fps, actual_half)
    """
    # Load model
    model = YOLO(model_name_or_path)

    # Handle FP16 check safely
    actual_half = half
    if half:
        if torch.cuda.is_available():
            model.to('cuda')
        else:
            # Fall back to standard or catch error
            actual_half = False

    # Detect frame source (webcam vs synthetic fallback)
    use_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap = None

    if not use_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            use_synthetic = True
            if cap is not None:
                cap.release()
                cap = None

    # Prepare frames
    frames = []
    if use_synthetic:
        # Pre-generate synthetic frames outside timed loop to minimize frame acquisition overhead
        import numpy as np
        synthetic_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        frames = [synthetic_frame.copy() for _ in range(num_frames + 5)] # 5 warmup frames + main loop
    else:
        # Read from webcam
        for _ in range(num_frames + 5):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame_resized = cv2.resize(frame, (imgsz, imgsz))
            frames.append(frame_resized)
        cap.release()

    # If no frames could be read or pre-generated, protect against division by zero
    if len(frames) == 0:
        return 0.0, 0.0, actual_half

    # 5-frame warmup phase using same precision as main loop
    warmup_frames = frames[:5]
    main_frames = frames[5:5 + num_frames]

    # Run warmup
    for f in warmup_frames:
        model(f, imgsz=imgsz, half=actual_half, verbose=False)

    # Main timed inference loop using high-precision time.perf_counter()
    latencies = []

    for f in main_frames:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            model(f, imgsz=imgsz, half=actual_half, verbose=False)
        except RuntimeError as e:
            # Fallback on failure (e.g. half precision on unsupported hardware/CPU)
            if actual_half:
                actual_half = False
                model(f, imgsz=imgsz, half=False, verbose=False)
            else:
                raise e

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

    # Protect against empty main frames or zero division
    if len(latencies) == 0:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half


def save_summary(model_name, imgsz, precision, fps, avg_latency, observation):
    """
    Saves or updates a benchmark result in results/tables/summary.csv.
    Ensures standard schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation.

    Args:
        model_name (str): Model variant (e.g., 'yolov8n').
        imgsz (int): Input size (e.g., 640).
        precision (str): Precision label ('FP32' or 'FP16').
        fps (float): Measured average FPS.
        avg_latency (float): Measured average latency in ms.
        observation (str): Descriptive observation text.
    """
    # Normalize model name to YOLOv8 prefix with lowercase suffix (e.g. YOLOv8n)
    normalized_model = model_name
    if normalized_model.lower().startswith('yolov8'):
        suffix = normalized_model[6:].lower()
        normalized_model = f"YOLOv8{suffix}"
    else:
        normalized_model = normalized_model.capitalize()

    resolution_str = f"{imgsz}x{imgsz}"

    # Locate summary.csv using absolute path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Read existing summary or create new if not present
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Schema migration / data recovery for legacy summary.csv
    # If Model or Precision columns are missing, default them to 'YOLOv8n' and 'FP32' respectively
    if not df.empty:
        if 'Model' not in df.columns:
            df['Model'] = 'YOLOv8n'
        if 'Precision' not in df.columns:
            df['Precision'] = 'FP32'

    # Standard schema columns
    expected_cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    # If df is empty, create structure with expected columns
    if df.empty:
        df = pd.DataFrame(columns=expected_cols)
    else:
        # Reorder/select expected columns to be safe
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        df = df[expected_cols]

    # Check if entry already exists (by matching Resolution, Model, Precision)
    match_mask = (df['Resolution'] == resolution_str) & \
                 (df['Model'].str.lower() == normalized_model.lower()) & \
                 (df['Precision'].str.upper() == precision.upper())

    new_row = {
        "Resolution": resolution_str,
        "Model": normalized_model,
        "Precision": precision.upper(),
        "Average_FPS": round(fps, 1),
        "Average_Latency_ms": round(avg_latency, 1),
        "Observation": observation
    }

    if match_mask.any():
        # Update existing record
        idx = df[match_mask].index[0]
        for col, val in new_row.items():
            df.at[idx, col] = val
    else:
        # Append new record
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Ensure correct column ordering before saving
    df = df[expected_cols]
    df.to_csv(summary_path, index=False)
    print(f"Results saved successfully to {summary_path}")
