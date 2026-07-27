import os
import sys

# Insert containing directory into sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Running Model Size Test (YOLOv8n vs YOLOv8s)...")

    # YOLOv8n at 640
    avg_latency_n, fps_n, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_n, avg_latency_n, "Ultra-lightweight, high FPS")

    # YOLOv8s at 640
    avg_latency_s, fps_s, _ = benchmark_model("yolov8s.pt", 640, "FP32")
    save_summary(640, "yolov8s.pt", "FP32", fps_s, avg_latency_s, "Balanced speed and accuracy")

    print("Model Size Test Complete.")

if __name__ == '__main__':
    run_model_size_test()
