import os
import sys
from utils import benchmark_model, save_summary

def main():
    print("Running Resolution Experiments...")

    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps, avg_latency, observation)

        print(f"Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
