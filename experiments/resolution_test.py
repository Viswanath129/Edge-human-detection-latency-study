import os
import sys
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model = YOLO("yolov8n.pt")

    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, actual_half = benchmark_model(model, imgsz=res, half=False)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )
        print(f"Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    run_test()
