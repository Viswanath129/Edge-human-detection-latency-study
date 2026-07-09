import os
import torch
from utils import benchmark_model, save_summary

def main():
    model_path = "yolov8n.pt"
    res = 640

    # Test FP32
    print("Benchmarking Precision: FP32")
    avg_latency_32, fps_32, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(
        resolution=f"{res}x{res}",
        model_name="yolov8n",
        precision="FP32",
        fps=fps_32,
        latency=avg_latency_32,
        observation="Standard precision"
    )
    print(f"FP32 Results: {fps_32:.2f} FPS, {avg_latency_32:.2f} ms")

    # Test FP16 (only if CUDA is available)
    if torch.cuda.is_available():
        print("Benchmarking Precision: FP16")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_path, imgsz=res, half=True)

        if actual_half:
            save_summary(
                resolution=f"{res}x{res}",
                model_name="yolov8n",
                precision="FP16",
                fps=fps_16,
                latency=avg_latency_16,
                observation="Hardware accelerated"
            )
            print(f"FP16 Results: {fps_16:.2f} FPS, {avg_latency_16:.2f} ms")
        else:
            print("FP16 benchmarking failed to use half-precision.")
    else:
        print("Skipping FP16 benchmark: CUDA not available. FP16 on CPU is often slower than FP32 and non-representative.")

if __name__ == "__main__":
    main()
