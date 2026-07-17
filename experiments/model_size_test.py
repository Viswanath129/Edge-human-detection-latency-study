import os
import sys

# Ensure local imports are resolvable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import benchmark_model, save_summary

def main():
    print("Starting Model Size Comparison Experiments...")

    resolution = 640
    precision = "FP32"

    # 1. Benchmark YOLOv8n (nano)
    nano_model = "yolov8n.pt"
    print(f"\n--- Benchmarking {nano_model} (Nano) ---")
    avg_latency_n, fps_n, _ = benchmark_model(nano_model, resolution=resolution, precision=precision)
    print(f"YOLOv8n results -> Latency: {avg_latency_n:.2f} ms, FPS: {fps_n:.2f}")
    save_summary(
        resolution_int_or_str=resolution,
        model_name=nano_model,
        precision=precision,
        avg_fps=fps_n,
        avg_latency_ms=avg_latency_n,
        observation="Lightweight nano model"
    )

    # 2. Benchmark YOLOv8s (small)
    small_model = "yolov8s.pt"
    print(f"\n--- Benchmarking {small_model} (Small) ---")
    avg_latency_s, fps_s, _ = benchmark_model(small_model, resolution=resolution, precision=precision)
    print(f"YOLOv8s results -> Latency: {avg_latency_s:.2f} ms, FPS: {fps_s:.2f}")
    save_summary(
        resolution_int_or_str=resolution,
        model_name=small_model,
        precision=precision,
        avg_fps=fps_s,
        avg_latency_ms=avg_latency_s,
        observation="Standard small model"
    )

    print("\nModel Size Comparison Experiments Completed Successfully!")

if __name__ == "__main__":
    main()
