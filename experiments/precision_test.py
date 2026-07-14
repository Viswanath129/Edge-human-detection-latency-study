import os
import torch
from utils import benchmark_model, save_summary

def run_precision_test():
    model_path = "yolov8n.pt"
    resolution = "640x640"
    imgsz = 640
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    print(f"Starting precision comparison for YOLOv8n at {resolution}...")

    # FP32
    print("Benchmarking FP32...")
    avg_latency_32, fps_32, _ = benchmark_model(
        model_path,
        imgsz=imgsz,
        half=False,
        force_synthetic=force_synthetic
    )
    save_summary(resolution, "yolov8n", "FP32", fps_32, avg_latency_32, "Standard precision")

    # FP16
    if torch.cuda.is_available():
        print("Benchmarking FP16...")
        avg_latency_16, fps_16, actual_half = benchmark_model(
            model_path,
            imgsz=imgsz,
            half=True,
            force_synthetic=force_synthetic
        )
        if actual_half:
            save_summary(resolution, "yolov8n", "FP16", fps_16, avg_latency_16, "Hardware accelerated")
        else:
            print("FP16 requested but not supported by hardware. Skipping update.")
    else:
        print("CUDA not available. Skipping FP16 benchmark.")

if __name__ == "__main__":
    run_precision_test()
