import os
import sys
import torch

# Ensure experiments directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("--- Running Precision Test ---")
    model_name = "yolov8n.pt"
    resolution = 640

    # Benchmark FP32
    print("Benchmarking FP32 precision...")
    avg_latency_fp32, fps_fp32, _ = benchmark_model(model_name, resolution, "FP32")
    save_summary(resolution, model_name, "FP32", fps_fp32, avg_latency_fp32, "Standard precision")

    # Benchmark FP16
    print("Benchmarking FP16 precision...")
    avg_latency_fp16, fps_fp16, actual_half = benchmark_model(model_name, resolution, "FP16")

    # If CUDA is unavailable, skip updating summary.csv for FP16
    if torch.cuda.is_available():
        save_summary(resolution, model_name, "FP16", fps_fp16, avg_latency_fp16, "Hardware-accelerated half precision")
    else:
        print("Skipping FP16 summary update because CUDA acceleration is unavailable.")

    print("Precision Test Completed.")

if __name__ == "__main__":
    main()
