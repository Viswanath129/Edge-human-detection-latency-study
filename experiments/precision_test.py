import sys
import os
import torch
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model_name = "yolov8n"
    model = YOLO(f"{model_name}.pt")
    size = 640

    # FP32
    print(f"Benchmarking {model_name} at {size}x{size} (FP32)")
    avg_latency, fps, _ = benchmark_model(model, imgsz=size, half=False)
    save_summary(f"{size}x{size}", model_name, "FP32", fps, avg_latency, "Standard precision")

    # FP16 (only if CUDA is available)
    if torch.cuda.is_available():
        print(f"Benchmarking {model_name} at {size}x{size} (FP16)")
        avg_latency, fps, actual_half = benchmark_model(model, imgsz=size, half=True)
        if actual_half:
            save_summary(f"{size}x{size}", model_name, "FP16", fps, avg_latency, "Half precision (GPU)")
    else:
        print("Skipping FP16 test as CUDA is not available.")

if __name__ == "__main__":
    run_test()
