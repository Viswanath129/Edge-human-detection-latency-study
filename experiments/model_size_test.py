import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Running Model Size Variant Test...")
    resolution = 640

    # YOLOv8n (nano)
    nano_model = "yolov8n.pt"
    avg_latency_n, fps_n, _ = benchmark_model(nano_model, resolution=resolution, half=False)
    print(f"YOLOv8n - Avg Latency: {avg_latency_n:.2f} ms, FPS: {fps_n:.2f}")
    save_summary(resolution, nano_model, "FP32", fps_n, avg_latency_n, "Ultra-lightweight edge model")

    # YOLOv8s (small)
    small_model = "yolov8s.pt"
    avg_latency_s, fps_s, _ = benchmark_model(small_model, resolution=resolution, half=False)
    print(f"YOLOv8s - Avg Latency: {avg_latency_s:.2f} ms, FPS: {fps_s:.2f}")
    save_summary(resolution, small_model, "FP32", fps_s, avg_latency_s, "Improved accuracy at latency cost")

if __name__ == "__main__":
    run_model_size_test()
