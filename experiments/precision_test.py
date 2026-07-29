import os
import sys
import torch

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Running Precision Test (YOLOv8n at 640x640)...")
    model_name = "yolov8n"
    resolution = 640

    # FP32 Benchmark
    print("Benchmarking at FP32 precision...")
    avg_latency_32, fps_32, _ = benchmark_model(model_name, resolution, "FP32", num_frames=50)
    print(f"FP32 - Avg Latency: {avg_latency_32:.2f} ms, FPS: {fps_32:.2f}")
    save_summary(resolution, model_name, "FP32", fps_32, avg_latency_32, "Standard Precision")

    # FP16 Benchmark
    print("Benchmarking at FP16 precision...")
    avg_latency_16, fps_16, actual_half = benchmark_model(model_name, resolution, "FP16", num_frames=50)
    print(f"FP16 - Avg Latency: {avg_latency_16:.2f} ms, FPS: {fps_16:.2f}")

    if not torch.cuda.is_available():
        print("CUDA acceleration is unavailable. Skipping updating summary.csv for FP16.")
    else:
        save_summary(resolution, model_name, "FP16", fps_16, avg_latency_16, "Reduced Precision (GPU Accelerated)")
    print("Precision Test completed.")

if __name__ == '__main__':
    main()
