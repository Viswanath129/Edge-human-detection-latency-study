import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_latency, fps, is_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if is_half else "FP32"
        obs = "Higher detection quality" if res == 640 else "Faster Inference"

        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_resolution_test()
