import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def normalize_model_name(model_name: str) -> str:
    """
    Normalizes model names to a standard 'YOLOv8' prefix with a lowercase suffix.
    E.g., 'yolov8n' or 'yolov8n.pt' becomes 'YOLOv8n'.
    """
    name = os.path.basename(model_name).strip()
    if name.lower().endswith(".pt"):
        name = name[:-3]
    if name.lower().startswith("yolov8"):
        suffix = name[6:].lower()
        return f"YOLOv8{suffix}"
    return name

def format_resolution(resolution) -> str:
    """
    Formats resolution to a standardized string 'WidthxHeight'.
    E.g., 640 or '640' becomes '640x640'.
    """
    res_str = str(resolution).strip()
    if "x" in res_str:
        return res_str
    return f"{res_str}x{res_str}"

def save_summary(resolution, model_name: str, precision, avg_fps: float, avg_latency: float, observation: str):
    """
    Consolidates benchmark results into the central results/tables/summary.csv file.
    Performs updates on existing matching entries instead of duplicate appends.
    Supports smooth schema migration by filling in missing legacy columns.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))

    schema_cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    # Read existing summary or create a new DataFrame
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=schema_cols)
    else:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        df = pd.DataFrame(columns=schema_cols)

    # Schema Migration: Populate missing 'Model' or 'Precision' columns with legacy defaults
    if "Model" not in df.columns:
        df["Model"] = "YOLOv8n"
    if "Precision" not in df.columns:
        df["Precision"] = "FP32"

    # Reorder/ensure all standard schema columns are present
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None
    df = df[schema_cols]

    # Normalize values for lookup
    formatted_res = format_resolution(resolution)
    norm_model = normalize_model_name(model_name)

    if isinstance(precision, bool):
        precision_str = "FP16" if precision else "FP32"
    else:
        precision_str = str(precision).strip().upper()

    # Search for an existing matching entry
    match_condition = (df["Resolution"] == formatted_res) & (df["Model"] == norm_model) & (df["Precision"] == precision_str)
    match_idx = df[match_condition].index

    if not match_idx.empty:
        # Update existing entry
        df.loc[match_idx[0], "Average_FPS"] = round(avg_fps, 2)
        df.loc[match_idx[0], "Average_Latency_ms"] = round(avg_latency, 2)
        df.loc[match_idx[0], "Observation"] = observation
    else:
        # Append new entry
        new_row = pd.DataFrame([{
            "Resolution": formatted_res,
            "Model": norm_model,
            "Precision": precision_str,
            "Average_FPS": round(avg_fps, 2),
            "Average_Latency_ms": round(avg_latency, 2),
            "Observation": observation
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(summary_path, index=False)

def benchmark_model(model_name: str, resolution: int, half: bool = False, force_synthetic: bool = False, num_frames: int = 50):
    """
    Benchmarks YOLO model inference speed at a given resolution.
    Returns (avg_latency_ms, fps, actual_half)
    """
    # Load model via ultralytics API
    model = YOLO(model_name)

    cuda_available = torch.cuda.is_available()
    actual_half = False

    # Safe FP16 check to prevent crashes on CPU
    if half:
        if cuda_available:
            actual_half = True
        else:
            # CPU check fallback
            try:
                # Test FP16 support on CPU with a small dummy inference
                dummy = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                _ = model(dummy, imgsz=resolution, half=True, verbose=False)
                actual_half = True
            except (RuntimeError, ValueError):
                half = False
                actual_half = False

    # Warmup phase: 5 frames using the same precision setting (FP16/FP32)
    warmup_frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(warmup_frame, imgsz=resolution, half=half, verbose=False)

    # Frame source logic: physical webcam vs synthetic NumPy frames
    synthetic_env = os.environ.get("FORCE_SYNTHETIC", "").lower() in ("true", "1")
    use_synthetic = force_synthetic or synthetic_env

    cap = None
    if not use_synthetic:
        cap = cv2.VideoCapture(0)
        if cap is not None and not cap.isOpened():
            cap.release()
            cap = None

    latencies = []

    if cap is not None:
        # Physical webcam benchmarking loop
        frames_processed = 0
        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (resolution, resolution))

            if cuda_available:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(frame_resized, imgsz=resolution, half=half, verbose=False)

            if cuda_available:
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            frames_processed += 1

        cap.release()
    else:
        # Pre-generate synthetic frames outside the timed loop to minimize overhead
        synthetic_frames = [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(num_frames)]

        # Pure inference timing loop
        for frame in synthetic_frames:
            if cuda_available:
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(frame, imgsz=resolution, half=half, verbose=False)

            if cuda_available:
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    # Hardened against division-by-zero errors
    if not latencies:
        return (0.0, 0.0, actual_half)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return (avg_latency, fps, actual_half)
