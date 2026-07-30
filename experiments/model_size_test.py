import os
import sys

# Ensure experiments directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("--- Running Model Size Test ---")
    resolution = 640

    # Nano model
    print("Benchmarking YOLOv8n (nano) model...")
    avg_latency_n, fps_n, _ = benchmark_model("yolov8n.pt", resolution, "FP32")
    save_summary(resolution, "yolov8n.pt", "FP32", fps_n, avg_latency_n, "Nano model - optimized for edge")

    # Small model
    print("Benchmarking YOLOv8s (small) model...")
    avg_latency_s, fps_s, _ = benchmark_model("yolov8s.pt", resolution, "FP32")
    save_summary(resolution, "yolov8s.pt", "FP32", fps_s, avg_latency_s, "Small model - balance of speed and accuracy")

    print("Model Size Test Completed.")

if __name__ == "__main__":
    main()
