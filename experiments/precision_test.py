import os
import sys
import torch

# Add current directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    model_name = "yolov8n.pt"
    res = 640

    # Test FP32
    print(f"\n--- Testing Precision: FP32 ---")
    avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(
        resolution=f"{res}x{res}",
        model_name=model_name,
        precision="FP32",
        fps=fps,
        latency=avg_latency,
        observation="Standard precision"
    )

    # Test FP16 (only record if actually used)
    if torch.cuda.is_available():
        print(f"\n--- Testing Precision: FP16 ---")
        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=True)
        if is_half:
            save_summary(
                resolution=f"{res}x{res}",
                model_name=model_name,
                precision="FP16",
                fps=fps,
                latency=avg_latency,
                observation="Half precision (CUDA accelerated)"
            )
    else:
        print("Skipping FP16 summary update as CUDA is not available.")

if __name__ == "__main__":
    run_precision_test()
