import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Resolution Benchmark Test (640 vs 416)...")

    # 1. 640x640 Input Resolution
    avg_latency_640, fps_640, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_640, avg_latency_640, "Higher detection quality")

    # 2. 416x416 Input Resolution
    avg_latency_416, fps_416, _ = benchmark_model("yolov8n.pt", 416, "FP32")
    save_summary(416, "yolov8n.pt", "FP32", fps_416, avg_latency_416, "Faster Inference")

    print("Resolution Benchmark Test completed successfully.")

if __name__ == "__main__":
    main()
