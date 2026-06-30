import os
import sys
import torch

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model_name = "yolov8n.pt"
    imgsz = 640
    precisions = [False, True]  # False=FP32, True=FP16

    for half in precisions:
        prec_label = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {prec_label}")

        # If FP16 requested but no CUDA, skip to avoid misleading results
        if half and not torch.cuda.is_available():
            print("Skipping FP16 test - CUDA not available.")
            continue

        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=imgsz, half=half)

        obs = "Standard precision" if not actual_half else "Hardware accelerated half-precision"
        save_summary(f"{imgsz}x{imgsz}", model_name, prec_label, fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
