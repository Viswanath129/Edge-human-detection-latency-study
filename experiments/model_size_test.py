import os
import sys

# Ensure experiments folder is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Model Size Benchmark (YOLOv8n vs YOLOv8s)...")

    # YOLOv8n has already been benchmarked in resolution_test.py for 640x640,
    # but let's run it again or make sure we have a fresh measurement for consistency.
    print("Benchmarking YOLOv8n (nano variant)...")
    latency_nano, fps_nano, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(
        resolution_str="640x640",
        model_variant="YOLOv8n",
        precision_str="FP32",
        avg_fps=fps_nano,
        avg_latency=latency_nano,
        observation="Nano model variant with minimum compute requirements"
    )

    # Benchmark YOLOv8s at 640x640
    print("Benchmarking YOLOv8s (small variant)...")
    latency_small, fps_small, _ = benchmark_model("yolov8s.pt", 640, "FP32")
    save_summary(
        resolution_str="640x640",
        model_variant="YOLOv8s",
        precision_str="FP32",
        avg_fps=fps_small,
        avg_latency=latency_small,
        observation="Small model variant offering higher detection capacity"
    )

    print("Model Size Benchmark completed successfully!")

if __name__ == "__main__":
    main()
