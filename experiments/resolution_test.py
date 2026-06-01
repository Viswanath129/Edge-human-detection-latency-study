import os
import sys
from ultralytics import YOLO

# Add parent directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    model = YOLO("yolov8n.pt")
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        fps, latency = benchmark_model(model, input_size=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps, latency, obs)

        print(f"Resolution {res}: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
