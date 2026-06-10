import os
import sys
from utils import benchmark_model, save_summary

def main():
    print("Running Model Size Experiments (Nano vs Small)...")

    res = 640
    models = {
        "yolov8n.pt": "YOLOv8n",
        "yolov8s.pt": "YOLOv8s"
    }

    for model_path, model_name in models.items():
        print(f"Benchmarking model: {model_name}")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        observation = "Lightweight edge model" if "n" in model_path else "Balanced accuracy/speed"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, observation)

        print(f"Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
