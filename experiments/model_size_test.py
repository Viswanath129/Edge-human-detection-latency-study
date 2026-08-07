import os
import sys

# Insert containing directory to sys.path to allow running from repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Model Size Benchmark Test...")
    resolution = 640

    # We test multiple model sizes: YOLOv8n, YOLOv8s, YOLOv8m
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    observations = {
        "yolov8n.pt": "Ultra lightweight model, fast on edge",
        "yolov8s.pt": "Balanced accuracy and latency trade-off",
        "yolov8m.pt": "Higher accuracy but slower inference speed"
    }

    for model_name in models:
        print(f"Benchmarking model {model_name} at {resolution}x{resolution}...")
        avg_latency, fps, actual_half = benchmark_model(model_name, resolution, half=False)

        print(f"Result - Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

        # Save summary to results/tables/summary.csv
        save_summary(
            resolution=resolution,
            model_name=model_name,
            precision="FP32",
            avg_fps=fps,
            avg_latency=avg_latency,
            observation=observations[model_name]
        )

    print("Model Size Benchmark Test completed and results saved.")

if __name__ == "__main__":
    main()
