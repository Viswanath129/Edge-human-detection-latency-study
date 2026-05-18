import cv2
import time
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

def get_frame_source(force_synthetic=False):
    """Detects if a webcam is available, otherwise returns synthetic fallback."""
    if force_synthetic:
        return None, False

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, True
    else:
        # Avoid repeated warnings if possible, but OpenCV might still log to stderr
        return None, False

def run_benchmark(model_path, imgsz=640, half=False, max_frames=50, headless=True):
    """Generic benchmarking loop for a given model and configuration."""
    model = YOLO(model_path)

    # If we are in a headless environment, we might prefer to skip webcam check entirely
    # to avoid hung processes or slow timeouts in some CI systems.
    force_synthetic = os.environ.get("FORCE_SYNTHETIC", "false").lower() == "true"
    cap, use_webcam = get_frame_source(force_synthetic=force_synthetic)

    if not use_webcam and not force_synthetic:
        print("Webcam not detected. Falling back to synthetic frames.")

    latencies = []
    frame_count = 0

    # Warmup
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy_frame, imgsz=imgsz, half=half, verbose=False)

    start_time = time.time()
    while frame_count < max_frames:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                use_webcam = False
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        t0 = time.time()
        results = model(frame, imgsz=imgsz, half=half, verbose=False)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000)
        frame_count += 1

        if not headless and use_webcam:
            annotated = results[0].plot()
            cv2.imshow("Human Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap:
        cap.release()
    if not headless:
        cv2.destroyAllWindows()

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    # Use sum of latencies for a more representative "Inference FPS"
    total_inference_time_s = sum(latencies) / 1000.0
    fps = frame_count / total_inference_time_s if total_inference_time_s > 0 else 0

    return avg_latency, fps, latencies

def save_results(filename, summary_data, raw_latencies=None):
    """Saves benchmark results to the results/tables directory."""
    os.makedirs("results/tables", exist_ok=True)

    # Save summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join("results/tables", filename), index=False)

    # Save raw latencies if provided (preserving research value)
    if raw_latencies is not None:
        raw_filename = "raw_" + filename
        df_raw = pd.DataFrame(raw_latencies)
        df_raw.to_csv(os.path.join("results/tables", raw_filename), index=False)
