import os
import time
import numpy as np
import pandas as pd
import torch
import cv2
from ultralytics import YOLO

def benchmark_model(model_name: str, resolution: int, precision: str = "FP32", num_frames: int = 50) -> tuple:
    """
    Benchmarks a YOLO model variant at a specific input resolution and precision level.
    Returns: (avg_latency_ms, fps, actual_half_used)
    """
    print(f"Benchmarking {model_name} | Resolution: {resolution}x{resolution} | Precision: {precision}")

    # Check if GPU is available
    cuda_available = torch.cuda.is_available()
    half_precision = (precision.upper() == "FP16")

    # FP16 requires half() on model & inputs, typically for CUDA or NPU.
    # On CPU, PyTorch doesn't natively support CPU FP16 operators for many layers and raises RuntimeError.
    # We will handle FP16 gracefully.
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return 0.0, 0.0, False

    # Move model to appropriate device and set half-precision if requested
    if cuda_available:
        model.to('cuda')
        if half_precision:
            model.model.half()
            actual_half = True
        else:
            actual_half = False
    else:
        # Fallback for CPU
        if half_precision:
            print("Warning: FP16 precision requested but CUDA is not available. Standard CPU fallback.")
            # Standard PyTorch fallback: run FP32 or use try-except around half
            try:
                model.model.half()
                actual_half = True
            except RuntimeError:
                print("PyTorch model.half() failed on CPU. Falling back to FP32.")
                actual_half = False
        else:
            actual_half = False

    # Force synthetic if environment variable is set
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"

    cap = None
    use_synthetic = force_synthetic

    if not force_synthetic:
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            print("Webcam not available. Falling back to synthetic numpy frames.")
            use_synthetic = True
            if cap:
                cap.release()
                cap = None

    # Pre-generate synthetic frames if needed, outside the timed inference loop to minimize overhead
    if use_synthetic:
        # Generate a batch of synthetic frames
        # Human detection expects typical 3-channel images
        synthetic_frames = [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(num_frames)]
    else:
        # Read a reference frame to size, then pre-allocate or dynamically size
        pass

    # 1. Warmup Phase (5 frames) using target precision settings
    print("Running 5 warmup frames...")
    warmup_res = resolution
    for i in range(5):
        if use_synthetic:
            frame = np.zeros((warmup_res, warmup_res, 3), dtype=np.uint8)
        else:
            ret, r_frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(r_frame, (warmup_res, warmup_res))

        # Run inference once (no gradient)
        with torch.no_grad():
            _ = model(frame, imgsz=resolution, verbose=False)
            if cuda_available:
                torch.cuda.synchronize()

    # 2. Main Inference Loop
    print(f"Running {num_frames} timed inference frames...")
    latencies = []
    frames_processed = 0

    for idx in range(num_frames):
        if use_synthetic:
            frame = synthetic_frames[idx]
        else:
            ret, r_frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(r_frame, (resolution, resolution))

        # Capture start time strictly around inference
        if cuda_available:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model(frame, imgsz=resolution, verbose=False)
            if cuda_available:
                torch.cuda.synchronize()

        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)
        frames_processed += 1

    if not use_synthetic and cap:
        cap.release()

    # Division-by-zero protection
    if frames_processed == 0:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / frames_processed
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print(f"Benchmark results - Avg Latency: {avg_latency:.2f} ms | FPS: {fps:.2f}")
    return avg_latency, fps, actual_half


def save_summary(resolution: str, model: str, precision: str, fps: float, latency: float, observation: str):
    """
    Saves or updates a benchmark entry in the centralized summary.csv file.
    Performs key-based updates on Resolution, Model, and Precision to prevent duplication.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "..", "results", "tables", "summary.csv"))

    # Ensure directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Normalize model name prefix YOLOv8 and lowercase suffix (e.g., yolov8n -> YOLOv8n)
    normalized_model = model
    if model.lower().startswith('yolov8'):
        suffix = model[6:]
        normalized_model = f"YOLOv8{suffix.lower()}"

    new_row = {
        'Resolution': resolution,
        'Model': normalized_model,
        'Precision': precision.upper(),
        'Average_FPS': round(fps, 1),
        'Average_Latency_ms': round(latency, 1),
        'Observation': observation
    }

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)

            # Handle schema migration gracefully if legacy table has missing columns
            # Default missing columns 'Model' to 'YOLOv8n' and 'Precision' to 'FP32'
            if 'Model' not in df.columns:
                df['Model'] = 'YOLOv8n'
            if 'Precision' not in df.columns:
                df['Precision'] = 'FP32'

            # Reorder or cast columns to match expected schema
            df['Resolution'] = df['Resolution'].astype(str)
            df['Model'] = df['Model'].astype(str)
            df['Precision'] = df['Precision'].astype(str)

            # Look for existing match based on Resolution, Model, Precision keys
            match_mask = (
                (df['Resolution'] == resolution) &
                (df['Model'].str.lower() == normalized_model.lower()) &
                (df['Precision'].str.upper() == precision.upper())
            )

            if match_mask.any():
                # Update row
                df.loc[match_mask, 'Average_FPS'] = round(fps, 1)
                df.loc[match_mask, 'Average_Latency_ms'] = round(latency, 1)
                df.loc[match_mask, 'Observation'] = observation
                print(f"Updated existing entry in summary.csv for {resolution}, {normalized_model}, {precision}")
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Appended new entry to summary.csv for {resolution}, {normalized_model}, {precision}")

            # Standardize column ordering
            cols = ['Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms', 'Observation']
            df = df[cols]
            df.to_csv(summary_path, index=False)

        except Exception as e:
            print(f"Error reading/updating summary.csv: {e}. Writing new summary.csv.")
            df = pd.DataFrame([new_row])
            df.to_csv(summary_path, index=False)
    else:
        df = pd.DataFrame([new_row])
        df.to_csv(summary_path, index=False)
        print(f"Created summary.csv with initial entry for {resolution}, {normalized_model}, {precision}")
