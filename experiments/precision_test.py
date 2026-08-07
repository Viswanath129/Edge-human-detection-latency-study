import os
import sys
import torch

# Insert containing directory to sys.path to allow running from repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Precision Benchmark Test...")
    model_name = "yolov8n.pt"
    resolution = 640

    # 1. Benchmark standard FP32 precision
    print(f"Benchmarking {model_name} with FP32 precision...")
    avg_latency_32, fps_32, actual_half_32 = benchmark_model(model_name, resolution, half=False)
    print(f"FP32 Result - Avg Latency: {avg_latency_32:.2f} ms, FPS: {fps_32:.2f}")

    # Save standard FP32 results to summary.csv
    save_summary(
        resolution=resolution,
        model_name=model_name,
        precision="FP32",
        avg_fps=fps_32,
        avg_latency=avg_latency_32,
        observation="Standard FP32 precision"
    )

    # 2. Benchmark FP16 half precision
    print(f"Benchmarking {model_name} with FP16 precision...")
    avg_latency_16, fps_16, actual_half_16 = benchmark_model(model_name, resolution, half=True)
    print(f"FP16 Result - Avg Latency: {avg_latency_16:.2f} ms, FPS: {fps_16:.2f}, Actual FP16: {actual_half_16}")

    # Check if CUDA acceleration is available
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("Skipping summary.csv update for FP16 because CUDA acceleration is unavailable (prevents misleading FP32 results under FP16 label).")
    else:
        # Save FP16 results only if CUDA is available (meaning FP16 was actually utilized and representative)
        save_summary(
            resolution=resolution,
            model_name=model_name,
            precision="FP16",
            avg_fps=fps_16,
            avg_latency=avg_latency_16,
            observation="Hardware-accelerated FP16 half precision"
        )
        print("FP16 results successfully saved to summary.csv.")

    print("Precision Benchmark Test completed.")

if __name__ == "__main__":
    main()
