import os
import sys
import torch

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    # FP16 is only meaningful on GPU; on CPU it might be slower or unsupported
    # But for benchmarking purposes, we can try to test both if possible
    precisions = [False, True] # False = FP32, True = FP16
    model_path = "yolov8n.pt"
    res = 640

    for half in precisions:
        label = "FP16" if half else "FP32"
        print(f"\n--- Testing Precision: {label} ---")

        # If FP16 requested but no CUDA, utils.benchmark_model will warn and fallback
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res, half=half)

        actual_label = "FP16" if actual_half else "FP32"

        # Skip saving if we requested FP16 but got FP32 (to avoid duplicate entries with wrong labels)
        if half and not actual_half:
            print("Skipping summary update for FP16 (unavailable).")
            continue

        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision=actual_label,
            fps=fps,
            latency=avg_latency,
            observation=f"Precision test ({actual_label})"
        )

if __name__ == "__main__":
    run_precision_test()
