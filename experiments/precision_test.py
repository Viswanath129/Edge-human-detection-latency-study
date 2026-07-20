import os
import sys
import torch

# Ensure experiments folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    # 1. Benchmark 640x640 FP32 (which is standard)
    lat_fp32, fps_fp32, _ = benchmark_model(model_name="yolov8n.pt", resolution=640, precision="FP32")
    save_summary(
        resolution="640x640",
        model="yolov8n",
        precision="FP32",
        fps=fps_fp32,
        latency=lat_fp32,
        observation="Standard Precision"
    )

    # 2. Benchmark 640x640 FP16
    # Skip saving results to CSV if CUDA is unavailable, as CPU FP16 performance is highly non-representative
    # and we want to avoid polluting summary.csv with misleading metrics
    cuda_available = torch.cuda.is_available()

    lat_fp16, fps_fp16, actual_half = benchmark_model(model_name="yolov8n.pt", resolution=640, precision="FP16")

    if cuda_available and actual_half:
        save_summary(
            resolution="640x640",
            model="yolov8n",
            precision="FP16",
            fps=fps_fp16,
            latency=lat_fp16,
            observation="Hardware-Accelerated Half Precision"
        )
    else:
        print("Skipping summary.csv update for FP16 because CUDA/Hardware-acceleration is unavailable (avoiding misleading CPU FP16 metrics).")

if __name__ == "__main__":
    main()
