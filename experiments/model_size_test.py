import os
import sys

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    # Comparing nano and small models at 640x640
    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt"
    }
    res = 640

    for name, path in models.items():
        print(f"Benchmarking Model: {name}")
        avg_latency, fps, actual_half = benchmark_model(path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        obs = "Lightweight / Edge-optimized" if name == "YOLOv8n" else "Improved accuracy / Higher compute"

        save_summary(
            resolution=f"{res}x{res}",
            model_name=name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_model_size_test()
