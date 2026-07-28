import os
import sys

# Insert containing directory into sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Running Resolution Test (640x640 vs 416x416)...")

    # 640x640
    avg_latency_640, fps_640, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_640, avg_latency_640, "Higher detection quality")

    # 416x416
    avg_latency_416, fps_416, _ = benchmark_model("yolov8n.pt", 416, "FP32")
    save_summary(416, "yolov8n.pt", "FP32", fps_416, avg_latency_416, "Faster Inference")

    print("Resolution Test Complete.")

if __name__ == '__main__':
    run_resolution_test()
