import os
import sys
from ultralytics import YOLO

# Add parent directory to path to allow importing utils if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def main():
    model = YOLO("yolov8n.pt")

    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="YOLOv8n",
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    main()
