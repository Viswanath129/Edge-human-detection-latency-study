import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

def benchmark_model(model_name_or_path, resolution, precision='FP32', num_frames=50):
    """
    Benchmarks a YOLO model with specified input resolution and precision.
    Returns a tuple of (avg_latency_ms, inference_only_fps, actual_half).
    """
    # Determine FP16 vs FP32
    half = False
    if precision == 'FP16':
        if torch.cuda.is_available():
            half = True
        else:
            # CPU fallback or safe FP16 check to prevent CPU crashes
            half = False

    try:
        model = YOLO(model_name_or_path)
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        return (0.0, 0.0, False)

    # Frame source: Webcam (cv2.VideoCapture) vs synthetic numpy frames
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    use_webcam = False
    cap = None

    if not force_synthetic:
        try:
            cap = cv2.VideoCapture(0)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    use_webcam = True
                else:
                    cap.release()
                    cap = None
        except Exception:
            if cap:
                cap.release()
            cap = None

    # Pre-generate or pre-capture frames outside the timed inference loop
    frames = []
    if use_webcam and cap:
        for _ in range(num_frames + 5):
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (resolution, resolution))
                frames.append(frame_resized)
            else:
                break
        cap.release()

    # Fallback to synthetic if not enough frames from webcam
    needed = (num_frames + 5) - len(frames)
    if needed > 0:
        for _ in range(needed):
            # Pre-generate synthetic frame outside the timed loop to minimize overhead
            frames.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    warmup_frames = frames[:5]
    inference_frames = frames[5:]

    actual_half = half

    # Run warmup phase (5 frames) using the same precision settings
    for w_frame in warmup_frames:
        try:
            _ = model(w_frame, imgsz=resolution, half=actual_half, verbose=False)
        except RuntimeError:
            if actual_half:
                # Fallback to FP32 on CPU if FP16 RuntimeError occurs
                actual_half = False
                try:
                    _ = model(w_frame, imgsz=resolution, half=actual_half, verbose=False)
                except Exception:
                    pass
            else:
                pass
        except Exception:
            pass

    # Main inference loop
    latencies = []
    use_cuda = torch.cuda.is_available() and next(model.model.parameters()).is_cuda

    for frame in inference_frames:
        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            _ = model(frame, imgsz=resolution, half=actual_half, verbose=False)
        except RuntimeError as e:
            if actual_half:
                actual_half = False
                # Fallback run
                _ = model(frame, imgsz=resolution, half=actual_half, verbose=False)
            else:
                raise e

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

    # Division-by-zero protection and handling empty latencies
    if not latencies:
        return (0.0, 0.0, actual_half)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return (avg_latency, fps, actual_half)


def save_summary(resolution, model, precision, avg_fps, avg_latency, observation):
    """
    Saves/updates a benchmark result in the central results/tables/summary.csv file.
    Matches entries using Resolution, Model, and Precision to update instead of duplicate.
    """
    # Normalize model name: YOLOv8 prefix with lowercase suffix (e.g., yolov8n.pt -> YOLOv8n)
    m_lower = model.lower()
    if m_lower.endswith('.pt'):
        m_lower = m_lower[:-3]

    if m_lower.startswith('yolov8'):
        normalized_model = "YOLOv8" + m_lower[6:]
    elif m_lower.startswith('yolo'):
        normalized_model = "YOLO" + m_lower[4:]
    else:
        normalized_model = model

    # Location-agnostic absolute path resolution for summary.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, '../results/tables/summary.csv'))

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Standardized schema columns
    columns = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']

    # Read existing CSV or create a new DataFrame
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=columns)

        # Detect missing 'Model' or 'Precision' columns (schema migration)
        missing_cols = False
        if 'Model' not in df.columns:
            df['Model'] = 'YOLOv8n'
            missing_cols = True
        if 'Precision' not in df.columns:
            df['Precision'] = 'FP32'
            missing_cols = True

        # Re-align columns to the standard schema
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)

    # Normalize resolution string (e.g. 640 -> 640x640)
    res_str = str(resolution)
    if 'x' not in res_str:
        res_str = f"{res_str}x{res_str}"

    # Match existing entry
    match_idx = df[(df['Resolution'] == res_str) &
                   (df['Model'] == normalized_model) &
                   (df['Precision'] == precision)].index

    new_row = {
        'Resolution': res_str,
        'Model': normalized_model,
        'Precision': precision,
        'Average_FPS': round(avg_fps, 2),
        'Average_Latency_ms': round(avg_latency, 2),
        'Observation': observation
    }

    if not match_idx.empty:
        # Update existing entry
        df.loc[match_idx[0]] = new_row
    else:
        # Append new entry
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save to file
    df.to_csv(summary_path, index=False)
