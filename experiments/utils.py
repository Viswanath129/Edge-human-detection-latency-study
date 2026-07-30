import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name, resolution, precision="FP32", num_frames=50):
    """
    Benchmarks a YOLO model at a given resolution and precision.
    Returns:
        (avg_latency, fps, actual_half)
    """
    if num_frames <= 0:
        return 0.0, 0.0, False

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return 0.0, 0.0, False

    # Determine whether to use FP16 (half-precision)
    actual_half = False
    if precision == "FP16":
        if torch.cuda.is_available():
            actual_half = True
        else:
            print("CUDA is unavailable. Falling back to FP32 for CPU inference.")
            actual_half = False

    # Branching logic for video source
    use_synthetic = False
    frames = []

    if os.environ.get("FORCE_SYNTHETIC", "").lower() in ["true", "1"]:
        use_synthetic = True
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            use_synthetic = True
        else:
            # Try reading one frame
            ret, frame = cap.read()
            if not ret:
                use_synthetic = True
                cap.release()
            else:
                # Webcam is active, pre-acquire/pre-generate frames outside of the timed loop
                # We need num_frames + 5 (for warmup) frames
                frames.append(cv2.resize(frame, (resolution, resolution)))
                for _ in range(num_frames + 5 - 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.resize(frame, (resolution, resolution)))
                cap.release()

                # If we read fewer frames than expected, pad with the last frame
                if len(frames) < num_frames + 5:
                    last_frame = frames[-1] if len(frames) > 0 else np.zeros((resolution, resolution, 3), dtype=np.uint8)
                    while len(frames) < num_frames + 5:
                        frames.append(last_frame)

    if use_synthetic:
        # Pre-generate synthetic frames to minimize timing overhead
        frames = [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(num_frames + 5)]

    # Warmup phase: 5-frame warmup using same precision settings
    warmup_frames = frames[:5]
    test_frames = frames[5:]

    for wf in warmup_frames:
        try:
            _ = model(wf, imgsz=resolution, half=actual_half, conf=0.4, iou=0.5, verbose=False)
        except RuntimeError as e:
            if actual_half:
                print(f"FP16 failed with RuntimeError: {e}. Falling back to FP32.")
                actual_half = False
                try:
                    _ = model(wf, imgsz=resolution, half=False, conf=0.4, iou=0.5, verbose=False)
                except Exception:
                    return 0.0, 0.0, False
            else:
                print(f"Error during warmup: {e}")
                return 0.0, 0.0, False
        except Exception as e:
            print(f"Error during warmup: {e}")
            return 0.0, 0.0, False

    # Inference loop with timing
    latencies = []
    cuda_avail = torch.cuda.is_available()

    for tf in test_frames:
        if cuda_avail:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        try:
            _ = model(tf, imgsz=resolution, half=actual_half, conf=0.4, iou=0.5, verbose=False)
        except Exception as e:
            print(f"Inference error: {e}")
            continue

        if cuda_avail:
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

    if len(latencies) == 0:
        return 0.0, 0.0, actual_half

    avg_latency = sum(latencies) / len(latencies)
    total_time_sec = sum(latencies) / 1000.0
    fps = len(latencies) / total_time_sec if total_time_sec > 0.0 else 0.0

    return avg_latency, fps, actual_half


def save_summary(resolution, model_name, precision, avg_fps, avg_latency, observation):
    """
    Saves or updates benchmark metrics in results/tables/summary.csv.
    """
    # Normalize model name: e.g. yolov8n or yolov8n.pt -> YOLOv8n
    model_lower = model_name.lower().replace(".pt", "")
    if "yolov8" in model_lower:
        suffix = model_lower.replace("yolov8", "")
        model_norm = f"YOLOv8{suffix}"
    else:
        model_norm = model_name

    # Normalize resolution to string '640x640'
    if isinstance(resolution, int):
        resolution_str = f"{resolution}x{resolution}"
    elif isinstance(resolution, str) and "x" not in resolution:
        resolution_str = f"{resolution}x{resolution}"
    else:
        resolution_str = str(resolution)

    # Find the target summary.csv path using absolute path resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(base_dir, "..", "results", "tables", "summary.csv"))

    # Ensure results/tables/ directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # Schema configuration
    columns = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    # Perform schema migration for missing columns
    if "Model" not in df.columns:
        df["Model"] = "YOLOv8n"
    if "Precision" not in df.columns:
        df["Precision"] = "FP32"

    # Match and update or append
    match_idx = df[(df["Resolution"] == resolution_str) &
                   (df["Model"] == model_norm) &
                   (df["Precision"] == precision)].index

    new_row = {
        "Resolution": resolution_str,
        "Model": model_norm,
        "Precision": precision,
        "Average_FPS": round(float(avg_fps), 2),
        "Average_Latency_ms": round(float(avg_latency), 2),
        "Observation": observation
    }

    if not match_idx.empty:
        # Update existing row
        for col, val in new_row.items():
            df.at[match_idx[0], col] = val
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Ensure final schema order
    df = df[columns]

    # Save summary.csv
    df.to_csv(summary_path, index=False)
    print(f"Successfully saved entry to {summary_path}")
