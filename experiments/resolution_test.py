import sys
import os
from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Starting Resolution Benchmark...")

    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing Resolution: {res}x{res}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps, latency, observation)

        print(f"Result - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
