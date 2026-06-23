import os
import sys
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    # Test both nano and small models at 640x640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        print(f"Benchmarking model size: {model_name}")
        model = YOLO(model_path)

        avg_latency, fps, actual_half = benchmark_model(model, imgsz=640, half=False)

        obs = "Nano model (optimized for edge)" if "n" in model_name else "Small model (higher capacity)"
        save_summary(
            resolution="640x640",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )
        print(f"Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    run_test()
