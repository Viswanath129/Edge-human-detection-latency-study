import os
import sys
import pandas as pd
from utils import benchmark_model, save_summary

def run_model_size_test():
    print("="*50)
    print("Running Model Size Comparison Test (YOLOv8n vs YOLOv8s)...")
    print("="*50)

    models = ["yolov8n.pt", "yolov8s.pt"]
    imgsz = 640

    for model_file in models:
        # Extract model name without extension
        model_name = os.path.splitext(model_file)[0] # e.g. yolov8n

        print(f"\nBenchmarking {model_name} at resolution {imgsz}x{imgsz}...")

        avg_latency, fps, actual_half = benchmark_model(model_file, imgsz, half=False, num_frames=50)

        if "yolov8n" in model_name:
            obs = "Ultra-lightweight, extremely fast on edge devices"
        else:
            obs = "Higher capacity and accuracy, but increased latency"

        # Save to summary
        save_summary(
            model_name=model_name,
            imgsz=imgsz,
            precision="FP32",
            fps=fps,
            avg_latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_model_size_test()
