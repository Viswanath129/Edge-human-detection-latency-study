import os
import sys
from utils import benchmark_model, save_summary

def main():
    print("Running Precision Experiments (FP32 vs FP16)...")

    model_path = "yolov8n.pt"
    res = 640

    # FP32
    print("Benchmarking FP32...")
    avg_latency_32, fps_32, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps_32, avg_latency_32, "Standard precision")

    # FP16
    print("Benchmarking FP16...")
    avg_latency_16, fps_16, actual_half = benchmark_model(model_path, imgsz=res, half=True)
    label = "FP16" if actual_half else "FP16 (fallback to FP32)"
    save_summary(f"{res}x{res}", "YOLOv8n", label, fps_16, avg_latency_16, "Reduced precision for speed")

    print(f"FP32 - Latency: {avg_latency_32:.2f}ms, FPS: {fps_32:.2f}")
    print(f"{label} - Latency: {avg_latency_16:.2f}ms, FPS: {fps_16:.2f}")

if __name__ == "__main__":
    main()
