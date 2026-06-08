import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_tests():
    models = ["yolov8n.pt", "yolov8s.pt"]
    imgsz = 640

    for model_path in models:
        model_name = "YOLOv8n" if "yolov8n" in model_path else "YOLOv8s"
        print(f"\n--- Testing Model Size: {model_name} ---")
        avg_fps, avg_latency = benchmark_model(model_path, imgsz=imgsz)

        observation = "Ultra-lightweight" if "n" in model_path else "Balanced accuracy/speed"
        save_summary(
            resolution=f"{imgsz}x{imgsz}",
            model_name=model_name,
            precision="FP32",
            avg_fps=avg_fps,
            avg_latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_model_size_tests()
