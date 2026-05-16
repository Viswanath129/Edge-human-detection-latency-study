import cv2
import time
import pandas as pd
import numpy as np
import argparse
import os
from ultralytics import YOLO

def run_benchmark(model_name, input_size=640, num_frames=50):
    # Load model via identifier string
    print(f"Loading model: {model_name}")
    model = YOLO(f"{model_name}.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    use_synthetic = False

    if not cap.isOpened():
        print("Webcam not found. Falling back to synthetic frames.")
        use_synthetic = True

    latencies = []
    frame_count = 0
    start_time = time.time()

    while frame_count < num_frames:
        if use_synthetic:
            frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            ret = True
        else:
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.time()

        # Run inference
        results = model(frame, conf=0.4, iou=0.5, verbose=False)

        t1 = time.time()
        latency = (t1 - t0) * 1000  # ms
        latencies.append(latency)

        frame_count += 1

    if not use_synthetic:
        cap.release()
        cv2.destroyAllWindows()

    end_time = time.time()

    # Metrics
    avg_latency = sum(latencies) / len(latencies)
    fps = frame_count / (end_time - start_time)

    print(f"Model: {model_name}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # Save results
    os.makedirs("results/tables", exist_ok=True)
    df = pd.DataFrame({
        "latency_ms": latencies
    })
    df.to_csv(f"results/tables/model_{model_name}_results.csv", index=False)

    return avg_latency, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n", help="Model variant (e.g., yolov8n, yolov8s)")
    args = parser.parse_args()

    run_benchmark(args.model)
