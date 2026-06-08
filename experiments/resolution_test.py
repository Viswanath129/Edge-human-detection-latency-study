import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_tests():
    model_path = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_fps, avg_latency = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="YOLOv8n",
            precision="FP32",
            avg_fps=avg_fps,
            avg_latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_resolution_tests()
