import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model_name = "yolov8n.pt"
    resolutions = [640, 416]

    for imgsz in resolutions:
        print(f"Benchmarking resolution: {imgsz}x{imgsz}")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=imgsz, half=False)

        obs = "Higher detection quality" if imgsz == 640 else "Faster Inference"
        save_summary(f"{imgsz}x{imgsz}", model_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
