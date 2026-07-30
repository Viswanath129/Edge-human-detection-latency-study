import os
import sys

# Ensure experiments directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("--- Running Resolution Test ---")
    model_name = "yolov8n.pt"

    # Benchmark 640x640
    print("Benchmarking 640x640 resolution...")
    avg_latency_640, fps_640, _ = benchmark_model(model_name, 640, "FP32")
    save_summary(640, model_name, "FP32", fps_640, avg_latency_640, "Higher detection quality")

    # Benchmark 416x416
    print("Benchmarking 416x416 resolution...")
    avg_latency_416, fps_416, _ = benchmark_model(model_name, 416, "FP32")
    save_summary(416, model_name, "FP32", fps_416, avg_latency_416, "Faster Inference")

    print("Resolution Test Completed.")

if __name__ == "__main__":
    main()
