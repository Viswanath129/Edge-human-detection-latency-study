import os
import sys
import torch

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    # Comparing FP32 and FP16 for YOLOv8n at 640x640
    model_path = "yolov8n.pt"
    res = 640

    # FP32
    print("Benchmarking Precision: FP32")
    avg_latency, fps, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(
        resolution=f"{res}x{res}",
        model_name="YOLOv8n",
        precision="FP32",
        fps=fps,
        latency=avg_latency,
        observation="Standard precision"
    )

    # FP16
    if torch.cuda.is_available():
        print("Benchmarking Precision: FP16")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res, half=True)
        if actual_half:
            save_summary(
                resolution=f"{res}x{res}",
                model_name="YOLOv8n",
                precision="FP16",
                fps=fps,
                latency=avg_latency,
                observation="Half precision (Hardware accelerated)"
            )
    else:
        print("Skipping FP16 benchmark: CUDA not available.")

if __name__ == "__main__":
    run_precision_test()
