import os
import sys
import torch

# Add the current directory to sys.path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    model_name = "yolov8n.pt"
    res = 640

    # FP32
    print(f"--- Benchmarking Precision: FP32 ---")
    avg_latency, fps, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, "Baseline precision")
    print(f"Results for FP32: {fps:.2f} FPS, {avg_latency:.2f} ms")

    # FP16 (if supported)
    if torch.cuda.is_available():
        print(f"--- Benchmarking Precision: FP16 ---")
        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=True)
        precision_str = "FP16" if actual_half else "FP32 (Fallback)"
        save_summary(f"{res}x{res}", model_name, precision_str, fps, avg_latency, "Hardware accelerated")
        print(f"Results for {precision_str}: {fps:.2f} FPS, {avg_latency:.2f} ms")
    else:
        print("CUDA not available. Skipping FP16 benchmark.")

if __name__ == "__main__":
    run_precision_test()
