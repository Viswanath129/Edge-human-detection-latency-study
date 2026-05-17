import cv2
import time
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO

def run_benchmark(model_variant="yolov8n.pt", input_size=640, num_frames=500, use_webcam=True):
    # Load model
    model = YOLO(model_variant)

    # Try to open webcam if requested
    cap = None
    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not available. Falling back to synthetic frames.")
            cap = None

    latencies = []
    frame_count = 0

    print(f"Starting benchmark for model {model_variant}...")

    start_time = time.time()

    for i in range(num_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate synthetic frame
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        t0 = time.time()

        # Resize frame
        frame_resized = cv2.resize(frame, (input_size, input_size))

        # Run inference
        results = model(frame_resized, conf=0.4, iou=0.5, verbose=False)

        t1 = time.time()
        latency = (t1 - t0) * 1000  # ms

        # Skip first few frames for warmup
        if i > 5:
            latencies.append(latency)

        frame_count += 1

    if cap:
        cap.release()
        cv2.destroyAllWindows()

    end_time = time.time()

    # Metrics
    if not latencies:
        return 0, 0

    avg_latency = sum(latencies) / len(latencies)
    fps = (frame_count - 6) / (end_time - start_time) if frame_count > 6 else frame_count / (end_time - start_time)

    print(f"Model: {model_variant}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # Save results
    os.makedirs("results/tables", exist_ok=True)
    df = pd.DataFrame({
        "latency_ms": latencies
    })
    df.to_csv(f"results/tables/model_{model_variant.split('.')[0]}_results.csv", index=False)

    return avg_latency, fps

if __name__ == "__main__":
    for variant in ["yolov8n.pt", "yolov8s.pt"]:
        run_benchmark(model_variant=variant, num_frames=50)
