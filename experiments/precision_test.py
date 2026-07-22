import os
import sys
import torch

# Ensure experiments folder is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Precision Benchmark (FP32 vs FP16)...")

    # 1. FP32 benchmark
    print("Benchmarking YOLOv8n at FP32...")
    latency_fp32, fps_fp32, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(
        resolution_str="640x640",
        model_variant="YOLOv8n",
        precision_str="FP32",
        avg_fps=fps_fp32,
        avg_latency=latency_fp32,
        observation="Standard full-precision floating point model"
    )

    # 2. FP16 benchmark
    # Note: If FP16 is requested but CUDA acceleration is unavailable,
    # we skip updating the summary CSV to prevent misleading FP32 results from being recorded under FP16.
    if not torch.cuda.is_available():
        print("Skipping summary.csv update for FP16 as CUDA acceleration is unavailable (running on CPU).")
        print("Obtaining FP16 metrics on CPU is non-representative and significantly slower.")

        # We can still run it locally for display/warning or skipped altogether.
        # Let's run it and show CPU stats, but do NOT save them under the FP16 label in summary.csv.
        latency_fp16, fps_fp16, actual_half = benchmark_model("yolov8n.pt", 640, "FP16")
        print(f"FP16 CPU run (fallback to FP32={not actual_half}): Latency={latency_fp16:.2f} ms, FPS={fps_fp16:.2f}")
    else:
        print("Benchmarking YOLOv8n at FP16 on GPU...")
        latency_fp16, fps_fp16, actual_half = benchmark_model("yolov8n.pt", 640, "FP16")
        if actual_half:
            save_summary(
                resolution_str="640x640",
                model_variant="YOLOv8n",
                precision_str="FP16",
                avg_fps=fps_fp16,
                avg_latency=latency_fp16,
                observation="Half-precision accelerated on compatible GPU hardware"
            )
        else:
            print("Warning: Requested FP16, but actual execution fell back to FP32. Skipping summary write.")

    print("Precision Benchmark completed!")

if __name__ == "__main__":
    main()
