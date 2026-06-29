import os
import sys

# Add current directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    model_name = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if is_half else "FP32"
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
