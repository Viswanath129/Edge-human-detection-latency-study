import sys
import os
from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Starting Model Size Benchmark...")

    models = {
        "yolov8n.pt": "YOLOv8n",
        "yolov8s.pt": "YOLOv8s"
    }
    res = 640

    for model_path, model_name in models.items():
        print(f"Testing Model: {model_name}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Ultra-lightweight" if model_name == "YOLOv8n" else "Improved accuracy, higher latency"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, latency, observation)

        print(f"Result - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
