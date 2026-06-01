import os
import sys
from ultralytics import YOLO

# Add parent directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    # Comparing Nano and Small
    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt"
    }
    res = 640

    for name, path in models.items():
        print(f"Benchmarking Model: {name}")
        model = YOLO(path)
        fps, latency = benchmark_model(model, input_size=res)

        obs = "Ultra-lightweight" if "n" in path else "Balanced performance"
        save_summary(f"{res}x{res}", name, "FP32", fps, latency, obs)
        print(f"{name}: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
