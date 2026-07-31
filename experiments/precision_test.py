import os
import sys
import torch

# Insert containing directory into sys.path to allow direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Precision Benchmark Test (FP32 vs FP16)...")

    # 1. FP32 Precision (Standard)
    avg_latency_32, fps_32, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_32, avg_latency_32, "Standard precision")

    # 2. FP16 Precision (Half)
    if torch.cuda.is_available():
        avg_latency_16, fps_16, actual_half = benchmark_model("yolov8n.pt", 640, "FP16")
        if actual_half:
            save_summary(640, "yolov8n.pt", "FP16", fps_16, avg_latency_16, "Half precision (FP16)")
        else:
            print("FP16 requested but could not be processed in half precision. Skipping CSV update.")
    else:
        print("CUDA is unavailable. Skipping updating summary.csv for FP16 precision to avoid recording misleading CPU FP32 performance under FP16 label.")

    print("Precision Benchmark Test completed.")

if __name__ == "__main__":
    main()
