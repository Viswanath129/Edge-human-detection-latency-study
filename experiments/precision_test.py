import os
import sys
import torch
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model = YOLO("yolov8n.pt")

    # Test FP32
    print("Benchmarking precision: FP32")
    avg_latency, fps, actual_half = benchmark_model(model, imgsz=640, half=False)
    save_summary(
        resolution="640x640",
        model_name="yolov8n",
        precision="FP32",
        fps=fps,
        latency=avg_latency,
        observation="Standard FP32 precision"
    )
    print(f"FP32 Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

    # Test FP16 (only update summary if hardware support is available)
    print("Benchmarking precision: FP16")
    avg_latency, fps, actual_half = benchmark_model(model, imgsz=640, half=True)

    if actual_half and torch.cuda.is_available():
        save_summary(
            resolution="640x640",
            model_name="yolov8n",
            precision="FP16",
            fps=fps,
            latency=avg_latency,
            observation="Half-precision inference (GPU accelerated)"
        )
        print(f"FP16 Result - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")
    else:
        print("Skipping FP16 summary update: FP16 not natively supported or no CUDA available.")

if __name__ == "__main__":
    run_test()
