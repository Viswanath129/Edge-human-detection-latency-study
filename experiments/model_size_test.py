import os
import sys
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Starting Model Size Test (Nano vs Small)...")

    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        print(f"Benchmarking model: {model_name}")

        model = YOLO(model_path)
        fps, latency = benchmark_model(model, imgsz=640)

        observation = "Highly optimized for edge" if "nano" in model_name or "n" in model_name else "Improved accuracy, higher latency"
        save_summary(
            resolution="640x640",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )
        print(f"Model {model_name} - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
