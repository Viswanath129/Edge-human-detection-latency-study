import os
import sys

# Ensure experiments folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    # 1. Benchmark YOLOv8n (Nano - already benchmarked, but let's run to ensure consistent hardware context)
    lat_nano, fps_nano, _ = benchmark_model(model_name="yolov8n.pt", resolution=640, precision="FP32")
    save_summary(
        resolution="640x640",
        model="yolov8n",
        precision="FP32",
        fps=fps_nano,
        latency=lat_nano,
        observation="Nano Model - Ultra lightweight"
    )

    # 2. Benchmark YOLOv8s (Small - larger, more parameters, higher accuracy)
    lat_small, fps_small, _ = benchmark_model(model_name="yolov8s.pt", resolution=640, precision="FP32")
    save_summary(
        resolution="640x640",
        model="yolov8s",
        precision="FP32",
        fps=fps_small,
        latency=lat_small,
        observation="Small Model - Improved accuracy trade-off"
    )

if __name__ == "__main__":
    main()
