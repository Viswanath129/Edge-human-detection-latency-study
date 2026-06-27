import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    res = 640
    model_name = "yolov8n.pt"

    # FP32
    print("Benchmarking Precision: FP32")
    avg_latency, fps, half = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, "Standard precision")

    # FP16 (only if CUDA is available, otherwise it's just a duplicate of FP32)
    if torch.cuda.is_available():
        print("Benchmarking Precision: FP16")
        avg_latency, fps, half = benchmark_model(model_name, imgsz=res, half=True)
        save_summary(f"{res}x{res}", "yolov8n", "FP16", fps, avg_latency, "Half precision (CUDA)")
    else:
        print("Skipping FP16 benchmark: CUDA not available.")

if __name__ == "__main__":
    run_precision_test()
