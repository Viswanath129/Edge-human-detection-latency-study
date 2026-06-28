import os
import sys

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"
    model_name = "YOLOv8n"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res, half=False)

        precision = "FP16" if actual_half else "FP32"
        observation = "Higher detection quality" if res == 640 else "Faster Inference"

        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_resolution_test()
