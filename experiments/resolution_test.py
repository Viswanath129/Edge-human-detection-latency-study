import os
import sys

# Ensure experiments folder is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Resolution Benchmark (640 vs 416)...")

    # 1. Benchmark 640x640
    print("Benchmarking resolution 640x640...")
    latency_640, fps_640, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(
        resolution_str="640x640",
        model_variant="YOLOv8n",
        precision_str="FP32",
        avg_fps=fps_640,
        avg_latency=latency_640,
        observation="Standard high resolution for detailed detection"
    )

    # 2. Benchmark 416x416
    print("Benchmarking resolution 416x416...")
    latency_416, fps_416, _ = benchmark_model("yolov8n.pt", 416, "FP32")
    save_summary(
        resolution_str="416x416",
        model_variant="YOLOv8n",
        precision_str="FP32",
        avg_fps=fps_416,
        avg_latency=latency_416,
        observation="Optimized resolution for low-latency edge deployment"
    )

    print("Resolution Benchmark completed successfully!")

if __name__ == "__main__":
    main()
