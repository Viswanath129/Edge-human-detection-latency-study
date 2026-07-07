import sys
import os
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model = YOLO("yolov8n.pt")

    for size in [640, 416]:
        print(f"Benchmarking resolution: {size}x{size}")
        avg_latency, fps, _ = benchmark_model(model, imgsz=size)

        obs = "Higher detection quality" if size == 640 else "Faster Inference"
        save_summary(f"{size}x{size}", "yolov8n", "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
