import os
import sys

# Add current directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"\n--- Testing Model Size: {model_name} ---")
        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if is_half else "FP32"
        observation = "Ultra-lightweight" if "yolov8n" in model_name else "Improved accuracy"

        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_model_size_test()
