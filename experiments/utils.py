import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def normalize_model_name(model_name):
    """
    Normalizes a model name to have the 'YOLOv8' prefix with a lowercase suffix.
    E.g., yolov8n.pt -> YOLOv8n, yolov8s -> YOLOv8s.
    """
    name = model_name
    if name.lower().endswith('.pt'):
        name = name[:-3]
    name_lower = name.lower()
    if name_lower.startswith('yolov8'):
        suffix = name_lower[6:]
        return f"YOLOv8{suffix}"
    return name

def format_resolution(res):
    """
    Formats resolution to a standardized 'WIDTHxHEIGHT' string.
    """
    if isinstance(res, int):
        return f"{res}x{res}"
    if isinstance(res, str) and "x" not in res:
        try:
            val = int(res)
            return f"{val}x{val}"
        except ValueError:
            pass
    return str(res)

def benchmark_model(model_name, resolution, half=False, num_frames=50):
    """
    Benchmarks a YOLO model at a given resolution.
    Returns: (avg_latency, fps, actual_half)
    """
    actual_half = half
    if actual_half:
        if not torch.cuda.is_available():
            actual_half = False

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return 0.0, 0.0, False

    # Frame source detection and acquisition
    use_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    frames = []
    total_needed = num_frames + 5

    if not use_synthetic:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                for _ in range(total_needed):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize to target resolution
                    frame_resized = cv2.resize(frame, (resolution, resolution))
                    frames.append(frame_resized)
                cap.release()
        except Exception:
            pass

        if len(frames) < total_needed:
            use_synthetic = True
            frames = []

    if use_synthetic:
        # Pre-generate synthetic frames to minimize frame acquisition overhead
        for _ in range(total_needed):
            frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
            frames.append(frame)

    if len(frames) < total_needed:
        return 0.0, 0.0, actual_half

    warmup_frames = frames[:5]
    main_frames = frames[5:]

    # Safe precision check with runtime fallback
    try:
        for frame in warmup_frames:
            model(frame, imgsz=resolution, half=actual_half, verbose=False)
    except RuntimeError as re:
        if actual_half:
            print(f"FP16 runtime error encountered. Falling back to FP32. Error: {re}")
            actual_half = False
            try:
                for frame in warmup_frames:
                    model(frame, imgsz=resolution, half=actual_half, verbose=False)
            except Exception:
                return 0.0, 0.0, False
        else:
            return 0.0, 0.0, False
    except Exception:
        return 0.0, 0.0, actual_half

    # Main inference loop using time.perf_counter()
    latencies = []
    try:
        for frame in main_frames:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            model(frame, imgsz=resolution, half=actual_half, verbose=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # ms
    except Exception as e:
        print(f"Error during main inference loop: {e}")
        return 0.0, 0.0, actual_half

    if not latencies:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    # Benchmarking FPS is calculated based strictly on inference latency (inference-only FPS)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return avg_latency, fps, actual_half

def save_summary(resolution, model_name, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates a benchmark entry in results/tables/summary.csv.
    Standardized schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "../results/tables/summary.csv"))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    normalized_model = normalize_model_name(model_name)
    formatted_res = format_resolution(resolution)

    columns = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    new_row = {
        "Resolution": formatted_res,
        "Model": normalized_model,
        "Precision": precision,
        "Average_FPS": round(avg_fps, 2),
        "Average_Latency_ms": round(avg_latency, 2),
        "Observation": observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)

            # Detect missing 'Model' or 'Precision' columns and populate legacy defaults
            if "Model" not in df.columns:
                df["Model"] = "YOLOv8n"
            if "Precision" not in df.columns:
                df["Precision"] = "FP32"

            for col in columns:
                if col not in df.columns:
                    df[col] = ""

            # Normalize values for clean matching
            df["Model"] = df["Model"].apply(normalize_model_name)
            df["Resolution"] = df["Resolution"].apply(format_resolution)

            # Identify existing entry by matching Resolution, Model, and Precision
            match_mask = (
                (df["Resolution"].astype(str) == str(formatted_res)) &
                (df["Model"].astype(str) == str(normalized_model)) &
                (df["Precision"].astype(str) == str(precision))
            )

            if match_mask.any():
                idx = df[match_mask].index[0]
                df.at[idx, "Average_FPS"] = round(avg_fps, 2)
                df.at[idx, "Average_Latency_ms"] = round(avg_latency, 2)
                df.at[idx, "Observation"] = observation
            else:
                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)

            df = df[columns]
        except Exception as e:
            print(f"Error updating summary.csv, starting fresh: {e}")
            df = pd.DataFrame([new_row], columns=columns)
    else:
        df = pd.DataFrame([new_row], columns=columns)

    df.to_csv(summary_path, index=False)
