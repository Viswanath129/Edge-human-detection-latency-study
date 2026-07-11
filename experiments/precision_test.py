import sys
import os
import torch

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    resolution = 640
    model_name = "yolov8n.pt"

    # FP32 Benchmark
    avg_latency_32, fps_32, _ = benchmark_model(model_name, imgsz=resolution, half=False)
    save_summary(resolution, model_name.replace(".pt", ""), "FP32", fps_32, avg_latency_32, "Standard precision")
    print(f"FP32: FPS: {fps_32:.2f}, Latency: {avg_latency_32:.2f} ms")

    # FP16 Benchmark (if available)
    if torch.cuda.is_available():
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name, imgsz=resolution, half=True)
        if actual_half:
            save_summary(resolution, model_name.replace(".pt", ""), "FP16", fps_16, avg_latency_16, "Half precision (accelerated)")
            print(f"FP16: FPS: {fps_16:.2f}, Latency: {avg_latency_16:.2f} ms")
    else:
        print("Skipping FP16 test as CUDA is not available.")

if __name__ == "__main__":
    run_precision_test()
