import os
import sys

# Ensure local imports are resolvable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils import benchmark_model, save_summary

def main():
    print("Starting Resolution Comparison Experiments...")

    model_name = "yolov8n.pt"
    precision = "FP32"

    # 1. Benchmark 640x640 resolution
    print("\n--- Benchmarking 640x640 input resolution ---")
    avg_latency_640, fps_640, _ = benchmark_model(model_name, resolution=640, precision=precision)
    print(f"640x640 results -> Latency: {avg_latency_640:.2f} ms, FPS: {fps_640:.2f}")
    save_summary(
        resolution_int_or_str=640,
        model_name=model_name,
        precision=precision,
        avg_fps=fps_640,
        avg_latency_ms=avg_latency_640,
        observation="Higher detection quality"
    )

    # 2. Benchmark 416x416 resolution
    print("\n--- Benchmarking 416x416 input resolution ---")
    avg_latency_416, fps_416, _ = benchmark_model(model_name, resolution=416, precision=precision)
    print(f"416x416 results -> Latency: {avg_latency_416:.2f} ms, FPS: {fps_416:.2f}")
    save_summary(
        resolution_int_or_str=416,
        model_name=model_name,
        precision=precision,
        avg_fps=fps_416,
        avg_latency_ms=avg_latency_416,
        observation="Faster Inference"
    )

    print("\nResolution Comparison Experiments Completed Successfully!")

if __name__ == "__main__":
    main()
