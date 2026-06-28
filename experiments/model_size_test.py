import os
import sys

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    # Comparing YOLOv8n (nano) and YOLOv8s (small)
    models = [
        {"path": "yolov8n.pt", "name": "YOLOv8n", "obs": "Fastest (Nano)"},
        {"path": "yolov8s.pt", "name": "YOLOv8s", "obs": "Better accuracy (Small)"}
    ]
    res = 640

    for m in models:
        print(f"Benchmarking model: {m['name']}")
        avg_latency, fps, actual_half = benchmark_model(m['path'], imgsz=res, half=False)

        precision = "FP16" if actual_half else "FP32"

        save_summary(
            resolution=f"{res}x{res}",
            model_name=m['name'],
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=m['obs']
        )

if __name__ == "__main__":
    run_model_size_test()
