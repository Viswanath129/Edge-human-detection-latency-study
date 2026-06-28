import os
import sys
import torch

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    model_path = "yolov8n.pt"
    model_name = "YOLOv8n"
    res = 640

    # FP32
    print("Benchmarking FP32 precision...")
    avg_latency_32, fps_32, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(
        resolution=f"{res}x{res}",
        model_name=model_name,
        precision="FP32",
        fps=fps_32,
        latency=avg_latency_32,
        observation="Standard precision"
    )

    # FP16
    if torch.cuda.is_available():
        print("Benchmarking FP16 precision...")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_path, imgsz=res, half=True)
        if actual_half:
            save_summary(
                resolution=f"{res}x{res}",
                model_name=model_name,
                precision="FP16",
                fps=fps_16,
                latency=avg_latency_16,
                observation="Hardware accelerated (FP16)"
            )
        else:
            print("FP16 requested but not active. Skipping update.")
    else:
        print("CUDA not available. Skipping FP16 benchmark as it is non-representative on CPU.")

if __name__ == "__main__":
    run_precision_test()
