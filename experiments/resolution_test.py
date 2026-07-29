import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Running Resolution Test (YOLOv8n, FP32)...")
    model_name = "yolov8n"
    precision = "FP32"

    # Benchmark 640x640 resolution
    print("Benchmarking at 640x640...")
    avg_latency_640, fps_640, _ = benchmark_model(model_name, 640, precision, num_frames=50)
    print(f"640x640 - Avg Latency: {avg_latency_640:.2f} ms, FPS: {fps_640:.2f}")
    save_summary(640, model_name, precision, fps_640, avg_latency_640, "Higher detection quality")

    # Benchmark 416x416 resolution
    print("Benchmarking at 416x416...")
    avg_latency_416, fps_416, _ = benchmark_model(model_name, 416, precision, num_frames=50)
    print(f"416x416 - Avg Latency: {avg_latency_416:.2f} ms, FPS: {fps_416:.2f}")
    save_summary(416, model_name, precision, fps_416, avg_latency_416, "Faster Inference")
    print("Resolution Test completed.")

if __name__ == '__main__':
    main()
