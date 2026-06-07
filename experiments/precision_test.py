import sys
import os
from utils import benchmark_model, save_summary

def run_precision_test():
    print("Starting Precision Benchmark...")

    model_path = "yolov8n.pt"
    res = 640
    precisions = ["FP32", "FP16"]

    for prec in precisions:
        print(f"Testing Precision: {prec}")
        fps, latency = benchmark_model(model_path, imgsz=res, precision=prec)

        observation = "Standard precision" if prec == "FP32" else "Optimized for GPU, slower on CPU"
        save_summary(f"{res}x{res}", "YOLOv8n", prec, fps, latency, observation)

        print(f"Result - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_precision_test()
