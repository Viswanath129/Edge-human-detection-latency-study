import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_path_or_name, resolution, precision, num_frames=50):
    """
    Benchmarks a YOLO model for inference latency and throughput (FPS).

    Parameters:
        model_path_or_name (str): Path or name of the YOLO model (e.g., 'yolov8n.pt').
        resolution (int): Input image resolution (e.g., 640 or 416).
        precision (str): Precision level ('FP32' or 'FP16').
        num_frames (int): Number of benchmark iterations.

    Returns:
        tuple: (avg_latency_ms, fps, actual_half)
    """
    # Initialize model
    try:
        model = YOLO(model_path_or_name)
    except Exception as e:
        print(f"Error loading model {model_path_or_name}: {e}")
        return 0.0, 0.0, False

    # Precision configuration and safe FP16 checking
    use_half = False
    actual_half = False
    if precision.upper() == "FP16":
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                use_half = True
                actual_half = True
            except Exception as e:
                print(f"CUDA initialization failed for FP16, falling back to FP32: {e}")
                use_half = False
                actual_half = False
        else:
            print("CUDA acceleration unavailable. FP16 requested but falling back to standard precision (FP32).")
            use_half = False
            actual_half = False

    # Frame source logic: Prioritize webcam unless FORCE_SYNTHETIC=true or camera fails
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    use_webcam = False
    cap = None

    if not force_synthetic:
        try:
            # Try to open default webcam
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    use_webcam = True
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            print(f"Webcam initialization failed: {e}. Falling back to synthetic frames.")
            cap = None

    # Handle synthetic fallback frame generation outside the loop
    synthetic_frames = []
    if not use_webcam:
        print("Using synthetic frame source for headless/repeatable benchmark.")
        # Pre-generate synthetic frames of shape (resolution, resolution, 3)
        # to minimize acquisition overhead and isolate inference-only performance
        for _ in range(max(num_frames, 5)):
            synthetic_frames.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    # 5-frame Warmup phase under identical precision settings
    warmup_frames = 5
    for i in range(warmup_frames):
        if use_webcam and cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            warmup_frame = frame
        else:
            warmup_frame = synthetic_frames[i % len(synthetic_frames)]

        try:
            model(warmup_frame, imgsz=resolution, half=use_half, verbose=False)
        except RuntimeError as e:
            # Handle potential runtime errors under half precision (e.g. CPU half-precision unsupported)
            if use_half:
                print(f"RuntimeError during FP16 warmup: {e}. Falling back to FP32 standard precision.")
                use_half = False
                actual_half = False
                # Re-try warmup with standard precision
                model(warmup_frame, imgsz=resolution, half=use_half, verbose=False)
            else:
                print(f"RuntimeError during warmup: {e}")
                if cap is not None:
                    cap.release()
                return 0.0, 0.0, actual_half

    # Main inference loop
    latencies = []

    for i in range(num_frames):
        if use_webcam and cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            inference_frame = frame
        else:
            inference_frame = synthetic_frames[i % len(synthetic_frames)]

        try:
            # Synchronize CUDA if active to get exact execution timing
            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            results = model(inference_frame, imgsz=resolution, half=use_half, verbose=False)

            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000.0  # ms
            latencies.append(latency)
        except RuntimeError as e:
            print(f"RuntimeError during inference iteration {i}: {e}")
            break

    # Clean up video capture
    if cap is not None:
        cap.release()

    # Guard against division-by-zero if no frames processed
    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def normalize_model_name(name):
    """
    Normalizes YOLO model names to standard 'YOLOv8' prefix with a lowercase suffix.
    E.g., 'yolov8n' or 'yolov8n.pt' becomes 'YOLOv8n'.
    """
    # Extract filename without path and extension
    base = os.path.basename(name)
    if base.endswith(".pt"):
        base = base[:-3]

    # Check if starts with yolov8 (case-insensitive)
    match = re.match(r"^yolov8([nsmldx]?)", base, re.IGNORECASE)
    if match:
        suffix = match.group(1).lower()
        return f"YOLOv8{suffix}"

    return base

def save_summary(resolution_int_or_str, model_name, precision, avg_fps, avg_latency_ms, observation):
    """
    Saves or updates a benchmark run result in the summary CSV.
    Handles absolute paths, legacy schema migrations, and updates matching runs.
    """
    # Normalize inputs
    res_str = str(resolution_int_or_str)
    if 'x' not in res_str:
        res_str = f"{res_str}x{res_str}"

    normalized_model = normalize_model_name(model_name)

    # Resolve absolute path relative to utils.py location
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(utils_dir, "..", "results", "tables", "summary.csv"))

    # Ensure tables directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Check if file exists, if not initialize with correct schema
    columns = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            # Schema migration: check for missing 'Model' or 'Precision' columns
            migrated = False
            if "Model" not in df.columns:
                df["Model"] = "YOLOv8n"  # default legacy value
                migrated = True
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"  # default legacy value
                migrated = True

            # Normalize existing model names in df for consistency
            df["Model"] = df["Model"].apply(normalize_model_name)

            # Reorder/ensure columns are standard
            df = df[columns]
        except Exception as e:
            print(f"Error reading existing summary.csv, re-initializing: {e}")
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    # Standardize types and trim spaces
    df["Resolution"] = df["Resolution"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Precision"] = df["Precision"].astype(str).str.strip()

    # Check if entry already exists (matching Resolution, Model, and Precision)
    match_mask = (df["Resolution"] == res_str) & (df["Model"] == normalized_model) & (df["Precision"] == precision)

    new_row = {
        "Resolution": res_str,
        "Model": normalized_model,
        "Precision": precision,
        "Average_FPS": round(float(avg_fps), 1),
        "Average_Latency_ms": round(float(avg_latency_ms), 1),
        "Observation": observation
    }

    if match_mask.any():
        # Update existing record
        idx = df[match_mask].index[0]
        df.at[idx, "Average_FPS"] = new_row["Average_FPS"]
        df.at[idx, "Average_Latency_ms"] = new_row["Average_Latency_ms"]
        df.at[idx, "Observation"] = new_row["Observation"]
    else:
        # Append new record
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

    # Write back to summary.csv
    try:
        df.to_csv(summary_path, index=False)
        print(f"Successfully saved run {res_str} - {normalized_model} - {precision} to {summary_path}")
    except Exception as e:
        print(f"Failed to write to summary.csv: {e}")
