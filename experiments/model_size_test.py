import sys
import os
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    resolutions = [640]
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        model = YOLO(model_path)

        for size in resolutions:
            print(f"Benchmarking model: {model_name} at {size}x{size}")
            avg_latency, fps, _ = benchmark_model(model, imgsz=size)

            obs = "Ultra-lightweight" if "nano" in model_name or "yolov8n" in model_name else "Balanced performance"
            save_summary(f"{size}x{size}", model_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
