import os
import sys
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Starting Resolution Test...")
    model = YOLO("yolov8n.pt")

    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        fps, latency = benchmark_model(model, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )
        print(f"Resolution {res}x{res} - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
