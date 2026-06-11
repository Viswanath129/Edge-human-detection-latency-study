import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]
    resolution = 640

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        print(f"Testing model size: {model_name}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=resolution)

        observation = "Lightweight for edge" if "n" in model_name else "Better accuracy, higher load"
        precision = "FP16" if actual_half else "FP32"

        save_summary(
            resolution=f"{resolution}x{resolution}",
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    main()
