import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name, resolution, precision, num_frames=50):
    """
    Benchmarks a YOLO model at a specific resolution and precision.
    Returns: (avg_latency, fps, actual_half)
    """
    # Load model
    if not model_name.endswith('.pt'):
        model_file = f"{model_name}.pt"
    else:
        model_file = model_name

    try:
        model = YOLO(model_file)
    except Exception:
        return (0.0, 0.0, False)

    # Determine precision setup
    actual_half = False
    if precision == 'FP16':
        if torch.cuda.is_available():
            actual_half = True
        else:
            actual_half = False

    # Check FORCE_SYNTHETIC env var
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() in ("true", "1")

    use_webcam = False
    cap = None
    if not force_synthetic:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                use_webcam = True
        except Exception:
            use_webcam = False

    frames_list = []
    if not use_webcam:
        # Pre-generate synthetic frames outside timed loop
        for _ in range(num_frames):
            frames_list.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    # Warmup phase: 5-frame warmup phase using the same precision settings
    warmup_frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for _ in range(5):
        try:
            _ = model(warmup_frame, imgsz=resolution, half=actual_half, verbose=False)
        except Exception:
            pass

    # Main benchmarking loop
    latencies = []

    if use_webcam:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (resolution, resolution))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _ = model(frame_resized, imgsz=resolution, half=actual_half, verbose=False)
            except RuntimeError:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        cap.release()
    else:
        for frame in frames_list:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _ = model(frame, imgsz=resolution, half=actual_half, verbose=False)
            except RuntimeError:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    if not latencies:
        return (0.0, 0.0, actual_half)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return (avg_latency, fps, actual_half)

def save_summary(resolution, model, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates benchmark metrics in results/tables/summary.csv.
    Normalizes model names and migrates legacy schema if needed.
    """
    # Normalize model name
    m_name = model.lower()
    if m_name.endswith('.pt'):
        m_name = m_name[:-3]
    if m_name.startswith('yolov8'):
        suffix = m_name[6:]
        normalized_model = f"YOLOv8{suffix}"
    else:
        normalized_model = model

    # Format resolution string
    if isinstance(resolution, int) or (isinstance(resolution, str) and 'x' not in resolution):
        res_str = f"{resolution}x{resolution}"
    else:
        res_str = str(resolution)

    # Resolve absolute path for summary.csv relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, '../results/tables/summary.csv'))

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Load or create summary dataframe
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])

        # Check and migrate legacy schema if Model or Precision is missing
        if 'Model' not in df.columns:
            df['Model'] = 'YOLOv8n'
        if 'Precision' not in df.columns:
            df['Precision'] = 'FP32'
    else:
        df = pd.DataFrame(columns=['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation'])

    # Standardize types for matching
    df['Resolution'] = df['Resolution'].astype(str)
    df['Model'] = df['Model'].astype(str)
    df['Precision'] = df['Precision'].astype(str)

    # Check for existing matching entry
    match_mask = (df['Resolution'] == res_str) & (df['Model'] == normalized_model) & (df['Precision'] == precision)

    if match_mask.any():
        df.loc[match_mask, 'Average_FPS'] = float(avg_fps)
        df.loc[match_mask, 'Average_Latency_ms'] = float(avg_latency)
        df.loc[match_mask, 'Observation'] = str(observation)
    else:
        new_row = pd.DataFrame([{
            'Resolution': res_str,
            'Model': normalized_model,
            'Precision': precision,
            'Average_FPS': float(avg_fps),
            'Average_Latency_ms': float(avg_latency),
            'Observation': str(observation)
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    # Ensure correct schema column ordering
    df = df[['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']]

    df.to_csv(summary_path, index=False)
