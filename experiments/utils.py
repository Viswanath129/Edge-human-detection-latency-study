import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def benchmark_model(model_name_or_path, resolution, precision="FP32", num_frames=50):
    """
    Benchmarks the specified YOLO model on either camera or synthetic frames.

    Args:
        model_name_or_path (str): Path or name of the YOLO model (e.g., 'yolov8n.pt').
        resolution (int): Resolution of input frames.
        precision (str): Precision setting, either 'FP32' or 'FP16'.
        num_frames (int): Number of frames to benchmark.

    Returns:
        tuple: (avg_latency_ms, fps, actual_half)
    """
    # Load model
    try:
        model = YOLO(model_name_or_path)
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        return (0.0, 0.0, False)

    # Determine precision and compatibility
    actual_half = False
    if precision.upper() == "FP16":
        if torch.cuda.is_available():
            actual_half = True
        else:
            print("CUDA is not available. Falling back to FP32.")
            actual_half = False

    # Frame source selection
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    use_synthetic = force_synthetic

    frames = []
    cap = None
    if not use_synthetic:
        try:
            # Try webcam
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                success, frame = cap.read()
                if success:
                    frames.append(frame)
                    # Pre-load/read remaining frames to eliminate capturing overhead from timing loop
                    for _ in range(num_frames - 1):
                        success, f = cap.read()
                        if success:
                            frames.append(f)
                        else:
                            break
                else:
                    use_synthetic = True
            else:
                use_synthetic = True
        except Exception as e:
            print(f"Webcam initialization failed: {e}. Falling back to synthetic frames.")
            use_synthetic = True
        finally:
            if cap is not None:
                cap.release()

    if use_synthetic or len(frames) == 0:
        # Pre-generate synthetic frames outside the timed inference loop to minimize overhead
        frames = [np.zeros((resolution, resolution, 3), dtype=np.uint8) for _ in range(num_frames)]

    # Hardened check against division-by-zero
    if len(frames) == 0:
        return (0.0, 0.0, actual_half)

    # Warmup phase (5 frames) using same precision settings as main inference loop
    warmup_frame = cv2.resize(frames[0], (resolution, resolution))
    for _ in range(5):
        try:
            _ = model(warmup_frame, imgsz=resolution, half=actual_half, verbose=False)
        except Exception:
            pass

    # Main inference loop
    latencies = []
    for f in frames:
        frame_resized = cv2.resize(f, (resolution, resolution))

        # CPU vs GPU synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            _ = model(frame_resized, imgsz=resolution, half=actual_half, verbose=False)
        except RuntimeError as e:
            # Fallback or CPU exception handling for FP16
            if actual_half:
                print(f"RuntimeError during half precision inference, falling back to FP32: {e}")
                actual_half = False
                _ = model(frame_resized, imgsz=resolution, half=False, verbose=False)
            else:
                raise e

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0) # ms

    if len(latencies) == 0:
        return (0.0, 0.0, actual_half)

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return (avg_latency, fps, actual_half)


def save_summary(resolution, model_name, precision, fps, latency, observation=""):
    """
    Saves or updates a benchmark run's summary metrics in the centralized CSV.

    Schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(utils_dir, "../results/tables/summary.csv"))

    # Format resolution to string
    if isinstance(resolution, int):
        res_str = f"{resolution}x{resolution}"
    else:
        res_str = str(resolution)

    # Normalize model name (e.g. yolov8n.pt -> YOLOv8n)
    name = model_name.lower().replace(".pt", "")
    if name.startswith("yolov8"):
        suffix = name[6:]
        normalized_model = "YOLOv8" + suffix
    else:
        normalized_model = model_name

    precision_str = str(precision).upper()

    # Read existing summary CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Handle schema migration gracefully
        if "Model" not in df.columns:
            df["Model"] = "YOLOv8n"
        if "Precision" not in df.columns:
            df["Precision"] = "FP32"
    else:
        df = pd.DataFrame(columns=["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"])

    # Ensure schema ordering and column list
    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    df = df.reindex(columns=cols)

    # Search for matching existing entry (Resolution, Model, Precision)
    match_mask = (df["Resolution"] == res_str) & (df["Model"] == normalized_model) & (df["Precision"] == precision_str)

    if match_mask.any():
        df.loc[match_mask, "Average_FPS"] = round(fps, 2)
        df.loc[match_mask, "Average_Latency_ms"] = round(latency, 2)
        if observation:
            df.loc[match_mask, "Observation"] = observation
    else:
        new_row = {
            "Resolution": res_str,
            "Model": normalized_model,
            "Precision": precision_str,
            "Average_FPS": round(fps, 2),
            "Average_Latency_ms": round(latency, 2),
            "Observation": observation
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Summary entry saved to {csv_path}: Resolution={res_str}, Model={normalized_model}, Precision={precision_str}")
