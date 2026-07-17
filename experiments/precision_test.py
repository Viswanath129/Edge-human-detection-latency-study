import os
import sys
import torch

# Ensure local imports are resolvable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import benchmark_model, save_summary

def main():
    print("Starting Precision Comparison Experiments...")

    model_name = "yolov8n.pt"
    resolution = 640

    # 1. Benchmark FP32 (standard precision)
    print("\n--- Benchmarking FP32 (Standard Precision) ---")
    avg_latency_32, fps_32, _ = benchmark_model(model_name, resolution=resolution, precision="FP32")
    print(f"FP32 results -> Latency: {avg_latency_32:.2f} ms, FPS: {fps_32:.2f}")
    save_summary(
        resolution_int_or_str=resolution,
        model_name=model_name,
        precision="FP32",
        avg_fps=fps_32,
        avg_latency_ms=avg_latency_32,
        observation="Standard FP32 precision"
    )

    # 2. Benchmark FP16 (half precision)
    # Skip updating summary.csv if FP16 is requested but CUDA is unavailable
    print("\n--- Benchmarking FP16 (Half Precision) ---")
    if torch.cuda.is_available():
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name, resolution=resolution, precision="FP16")
        if actual_half:
            print(f"FP16 results -> Latency: {avg_latency_16:.2f} ms, FPS: {fps_16:.2f}")
            save_summary(
                resolution_int_or_str=resolution,
                model_name=model_name,
                precision="FP16",
                avg_fps=fps_16,
                avg_latency_ms=avg_latency_16,
                observation="Accelerated FP16 half precision"
            )
        else:
            print("FP16 was requested but could not be run at half precision. Skipping summary update.")
    else:
        print("CUDA acceleration is unavailable in this environment.")
        print("Skipping FP16 benchmark and summary.csv update to prevent recording misleading FP32 results under the FP16 label.")

    print("\nPrecision Comparison Experiments Completed Successfully!")

if __name__ == "__main__":
    main()
