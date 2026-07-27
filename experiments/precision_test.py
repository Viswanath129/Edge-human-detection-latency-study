import os
import sys
import torch

# Insert containing directory into sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    print("Running Precision Test (FP32 vs FP16)...")

    # FP32 (Base)
    avg_latency_32, fps_32, _ = benchmark_model("yolov8n.pt", 640, "FP32")
    save_summary(640, "yolov8n.pt", "FP32", fps_32, avg_latency_32, "Standard baseline precision")

    # FP16 (Half)
    if not torch.cuda.is_available():
        print("CUDA is unavailable. Skipping FP16 summary update to avoid recording misleading CPU results under FP16.")
    else:
        avg_latency_16, fps_16, actual_half = benchmark_model("yolov8n.pt", 640, "FP16")
        if actual_half:
            save_summary(640, "yolov8n.pt", "FP16", fps_16, avg_latency_16, "Reduced precision, hardware accelerated")
        else:
            print("FP16 was requested but could not be run with half precision. Skipping summary update.")

    print("Precision Test Complete.")

if __name__ == '__main__':
    run_precision_test()
