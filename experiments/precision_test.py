import os
import sys
import torch
from utils import benchmark_model, save_summary

def run_precision_test():
    print("="*50)
    print("Running Precision Level Test (FP32 vs FP16)...")
    print("="*50)

    model_file = "yolov8n.pt"
    imgsz = 640
    precisions = [False, True]  # False is FP32, True is FP16

    for half in precisions:
        precision_label = "FP16" if half else "FP32"
        print(f"\nBenchmarking {model_file} with {precision_label} precision...")

        # Run benchmark
        avg_latency, fps, actual_half = benchmark_model(model_file, imgsz, half=half, num_frames=50)

        if half:
            # Skip updating summary.csv if FP16 is requested but CUDA is unavailable
            if not torch.cuda.is_available() or not actual_half:
                print("Skipping summary.csv update for FP16 because CUDA acceleration is unavailable.")
                continue
            obs = "FP16 reduced precision, accelerated on compatible GPU/NPU hardware"
        else:
            obs = "Standard single-precision floating point"

        # Save to summary if not skipped
        save_summary(
            model_name="yolov8n",
            imgsz=imgsz,
            precision=precision_label,
            fps=fps,
            avg_latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_precision_test()
