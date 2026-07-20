import os
import sys

# Ensure experiments folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    # 1. Benchmark 640x640 Input Resolution
    lat_640, fps_640, _ = benchmark_model(model_name="yolov8n.pt", resolution=640, precision="FP32")
    save_summary(
        resolution="640x640",
        model="yolov8n",
        precision="FP32",
        fps=fps_640,
        latency=lat_640,
        observation="Higher detection quality"
    )

    # 2. Benchmark 416x416 Input Resolution
    lat_416, fps_416, _ = benchmark_model(model_name="yolov8n.pt", resolution=416, precision="FP32")
    save_summary(
        resolution="416x416",
        model="yolov8n",
        precision="FP32",
        fps=fps_416,
        latency=lat_416,
        observation="Faster Inference"
    )

if __name__ == "__main__":
    main()
