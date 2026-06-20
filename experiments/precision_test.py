import sys
import os
import torch
from ultralytics import YOLO

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model = YOLO("yolov8n.pt")
    res = 640

    # FP32
    print("Benchmarking precision: FP32...")
    avg_latency_32, fps_32, _ = benchmark_model(model, imgsz=res, half=False)
    save_summary(
        resolution=f"{res}x{res}",
        model_name="YOLOv8n",
        precision="FP32",
        fps=fps_32,
        latency=avg_latency_32,
        observation="Standard precision"
    )
    print(f"FP32 | Avg Latency: {avg_latency_32:.2f} ms | FPS: {fps_32:.2f}")

    # FP16 (only if CUDA available)
    if torch.cuda.is_available():
        print("Benchmarking precision: FP16...")
        avg_latency_16, fps_16, actual_half = benchmark_model(model, imgsz=res, half=True)
        if actual_half:
            save_summary(
                resolution=f"{res}x{res}",
                model_name="YOLOv8n",
                precision="FP16",
                fps=fps_16,
                latency=avg_latency_16,
                observation="Half precision (CUDA)"
            )
            print(f"FP16 | Avg Latency: {avg_latency_16:.2f} ms | FPS: {fps_16:.2f}")
    else:
        print("FP16 skipped: CUDA not available")

if __name__ == "__main__":
    run_test()
