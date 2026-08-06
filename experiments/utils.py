import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def normalize_model_name(name):
    """
    Normalizes model names to a standard 'YOLOv8' prefix with a lowercase suffix.
    E.g., 'yolov8n' or 'yolov8n.pt' becomes 'YOLOv8n'.
    """
    name_str = str(name).replace(".pt", "")
    if name_str.lower().startswith("yolov8"):
        suffix = name_str[6:].lower()
        return f"YOLOv8{suffix}"
    return name_str

def benchmark_model(model_name_or_path, resolution=640, half=False, num_frames=50):
    """
    Benchmarks a YOLO model at a specific resolution and precision.
    Returns:
        (avg_latency, fps, actual_half)
    """
    # Load model
    model = YOLO(model_name_or_path)

    # Safe FP16 check using torch.cuda.is_available()
    # Falls back to standard precision or catches RuntimeError on CPU
    actual_half = half
    if actual_half and not torch.cuda.is_available():
        actual_half = False

    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() in ("true", "1", "yes")
    frames = []

    # Prioritize physical webcam unless FORCE_SYNTHETIC is set or camera fails
    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            for _ in range(num_frames + 5):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (resolution, resolution))
                frames.append(frame_resized)
            cap.release()

    # Fallback to pre-generated synthetic numpy frames outside the timed loop
    if len(frames) < num_frames + 5:
        frames = []
        for _ in range(num_frames + 5):
            synth_frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
            frames.append(synth_frame)

    # Division-by-zero hardening
    if not frames:
        return 0.0, 0.0, actual_half

    # Warmup phase: 5 frames using the same precision settings
    warmup_frames = frames[:5]
    inference_frames = frames[5:5+num_frames]

    use_half = actual_half
    try:
        for f in warmup_frames:
            _ = model(f, imgsz=resolution, half=use_half, conf=0.4, iou=0.5, verbose=False)
    except RuntimeError:
        if use_half:
            # Fallback to standard precision to prevent benchmarking crashes
            use_half = False
            for f in warmup_frames:
                _ = model(f, imgsz=resolution, half=use_half, conf=0.4, iou=0.5, verbose=False)
        else:
            return 0.0, 0.0, False

    actual_half = use_half

    # Timed inference loop using time.perf_counter()
    latencies = []
    cuda_available = torch.cuda.is_available()

    for f in inference_frames:
        if cuda_available:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(f, imgsz=resolution, half=actual_half, conf=0.4, iou=0.5, verbose=False)

        if cuda_available:
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates benchmark results in results/tables/summary.csv.
    """
    model_normalized = normalize_model_name(model)

    # Absolute path derived from its own file location to locate summary.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=cols)

        # Detect missing 'Model' or 'Precision' columns and populate with defaults
        missing_model = "Model" not in df.columns
        missing_precision = "Precision" not in df.columns

        if missing_model:
            df["Model"] = "YOLOv8n"
        if missing_precision:
            df["Precision"] = "FP32"

        for col in cols:
            if col not in df.columns:
                df[col] = ""
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)

    # Standardize resolution format (e.g., 640x640)
    if "x" not in str(resolution):
        res_str = f"{resolution}x{resolution}"
    else:
        res_str = str(resolution)

    precision_str = "FP16" if precision in (True, "FP16") else "FP32"

    # Identify existing entry by matching Resolution, Model, and Precision
    match_mask = (df["Resolution"] == res_str) & (df["Model"] == model_normalized) & (df["Precision"] == precision_str)

    new_row = {
        "Resolution": res_str,
        "Model": model_normalized,
        "Precision": precision_str,
        "Average_FPS": round(float(avg_fps), 2),
        "Average_Latency_ms": round(float(avg_latency), 2),
        "Observation": str(observation)
    }

    if match_mask.any():
        idx = df[match_mask].index[0]
        for col, val in new_row.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df[cols]
    df.to_csv(summary_path, index=False)
