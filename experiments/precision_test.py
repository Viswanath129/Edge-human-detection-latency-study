import os
import sys
import torch

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    model_name = "yolov8n.pt"
    res = 640

    # Test FP32
    print("Benchmarking precision: FP32")
    avg_latency_32, fps_32, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "yolov8n", "FP32", fps_32, avg_latency_32, "Standard precision")
    print(f"FP32 - Latency: {avg_latency_32:.2f}ms, FPS: {fps_32:.2f}")

    # Test FP16 (only if CUDA is available, otherwise it's just repeating FP32)
    if torch.cuda.is_available():
        print("Benchmarking precision: FP16")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", "yolov8n", "FP16", fps_16, avg_latency_16, "Hardware accelerated")
            print(f"FP16 - Latency: {avg_latency_16:.2f}ms, FPS: {fps_16:.2f}")
    else:
        print("Skipping FP16 benchmark as CUDA is not available (results would be identical to FP32 or slower on CPU).")

if __name__ == "__main__":
    main()
