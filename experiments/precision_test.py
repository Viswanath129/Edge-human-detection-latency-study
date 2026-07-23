import os
import sys
import torch

# Ensure the experiments directory is in the import path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Running Precision Experiments (FP32 vs FP16)...")

    model_path = "yolov8n.pt"
    res = 640

    # FP32 Benchmark
    print("Benchmarking FP32...")
    avg_latency_32, fps_32, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps_32, avg_latency_32, "Standard precision")
    print(f"FP32 - Latency: {avg_latency_32:.2f}ms, FPS: {fps_32:.2f}")

    # FP16 Benchmark (Skipped if CUDA is not available to prevent non-representative or CPU-slow benchmarks)
    if not torch.cuda.is_available():
        print("Skipping FP16 summary update because CUDA acceleration is unavailable (to avoid recording misleading performance results).")
    else:
        print("Benchmarking FP16...")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_path, imgsz=res, half=True)
        label = "FP16" if actual_half else "FP16 (fallback to FP32)"
        save_summary(f"{res}x{res}", "YOLOv8n", label, fps_16, avg_latency_16, "Reduced precision for speed")
        print(f"{label} - Latency: {avg_latency_16:.2f}ms, FPS: {fps_16:.2f}")

if __name__ == "__main__":
    main()
