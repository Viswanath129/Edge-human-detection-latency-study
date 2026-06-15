import os
import sys
from ultralytics import YOLO

# Add parent directory to path to allow importing utils if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_path in models:
        model_name = "YOLOv8n" if "n" in model_path else "YOLOv8s"
        print(f"Benchmarking model: {model_name}")
        model = YOLO(model_path)
        avg_latency, fps, _ = benchmark_model(model, imgsz=640)

        obs = "Ultra-lightweight" if "n" in model_path else "Balanced accuracy/speed"

        save_summary(
            resolution="640x640",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    main()
