import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Running Model Size Test (640x640, FP32)...")
    resolution = 640
    precision = "FP32"

    # YOLOv8n
    print("Benchmarking YOLOv8n...")
    avg_latency_n, fps_n, _ = benchmark_model("yolov8n", resolution, precision, num_frames=50)
    print(f"YOLOv8n - Avg Latency: {avg_latency_n:.2f} ms, FPS: {fps_n:.2f}")
    save_summary(resolution, "yolov8n", precision, fps_n, avg_latency_n, "Higher detection quality")

    # YOLOv8s
    print("Benchmarking YOLOv8s...")
    avg_latency_s, fps_s, _ = benchmark_model("yolov8s", resolution, precision, num_frames=50)
    print(f"YOLOv8s - Avg Latency: {avg_latency_s:.2f} ms, FPS: {fps_s:.2f}")
    save_summary(resolution, "yolov8s", precision, fps_s, avg_latency_s, "Larger capacity model")
    print("Model Size Test completed.")

if __name__ == '__main__':
    main()
