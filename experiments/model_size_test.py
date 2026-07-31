import os
import sys

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Model Size Benchmark Test (Nano vs Small)...")

    # 1. Nano variant (yolov8n.pt)
    avg_latency_n, fps_n, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_n, avg_latency_n, "Nano model - optimized for edge")

    # 2. Small variant (yolov8s.pt)
    # The pre-trained weights yolov8s.pt will be fetched automatically via Ultralytics API
    avg_latency_s, fps_s, _ = benchmark_model("yolov8s.pt", 640, "FP32")
    save_summary(640, "yolov8s.pt", "FP32", fps_s, avg_latency_s, "Small model - higher accuracy")

    print("Model Size Benchmark Test completed successfully.")

if __name__ == "__main__":
    main()
