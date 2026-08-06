import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Running Input Resolution Test...")
    model_name = "yolov8n.pt"

    # 640x640 Benchmark
    avg_latency_640, fps_640, _ = benchmark_model(model_name, resolution=640, half=False)
    print(f"640x640 - Avg Latency: {avg_latency_640:.2f} ms, FPS: {fps_640:.2f}")
    save_summary(640, model_name, "FP32", fps_640, avg_latency_640, "Higher detection quality")

    # 416x416 Benchmark
    avg_latency_416, fps_416, _ = benchmark_model(model_name, resolution=416, half=False)
    print(f"416x416 - Avg Latency: {avg_latency_416:.2f} ms, FPS: {fps_416:.2f}")
    save_summary(416, model_name, "FP32", fps_416, avg_latency_416, "Faster Inference")

if __name__ == "__main__":
    run_resolution_test()
