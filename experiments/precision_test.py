import os
import sys
import torch

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    print("Running Precision Level Test...")
    model_name = "yolov8n.pt"
    resolution = 640

    # FP32 Benchmark
    avg_latency_32, fps_32, _ = benchmark_model(model_name, resolution=resolution, half=False)
    print(f"FP32 - Avg Latency: {avg_latency_32:.2f} ms, FPS: {fps_32:.2f}")
    save_summary(resolution, model_name, "FP32", fps_32, avg_latency_32, "Standard Precision")

    # FP16 Benchmark
    # Skip updating summary.csv if FP16 is requested but CUDA acceleration is unavailable
    if not torch.cuda.is_available():
        print("CUDA acceleration unavailable. Skipping recording FP16 benchmark to prevent non-representative or CPU fallback measurements.")
    else:
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name, resolution=resolution, half=True)
        if actual_half:
            print(f"FP16 - Avg Latency: {avg_latency_16:.2f} ms, FPS: {fps_16:.2f}")
            save_summary(resolution, model_name, "FP16", fps_16, avg_latency_16, "Reduced precision accelerated")
        else:
            print("FP16 benchmarking fell back to FP32 or failed. Skipping update.")

if __name__ == "__main__":
    run_precision_test()
