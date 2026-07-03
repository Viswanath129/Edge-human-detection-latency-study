import os
import sys
import torch

# Ensure experiments directory is in path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    precisions = [False, True]  # FP32, FP16
    model_name = "yolov8n.pt"
    res = 640

    for half in precisions:
        label = "FP16" if half else "FP32"
        print(f"Testing precision: {label}...")

        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=half)

        if half and not actual_half:
            print("Skipping FP16 results as CUDA is not available.")
            continue

        precision = "FP16" if actual_half else "FP32"
        observation = "Standard FP32" if not actual_half else "Accelerated FP16"

        save_summary(f"{res}x{res}", fps, avg_latency, observation, "yolov8n", precision)

if __name__ == "__main__":
    run_precision_test()
