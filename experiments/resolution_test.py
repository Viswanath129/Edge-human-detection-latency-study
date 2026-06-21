import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_latency, fps, is_half = benchmark_model(model_name=model_name, imgsz=res, half=False)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=res,
            model_name=model_name.split('.')[0],
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_resolution_test()
