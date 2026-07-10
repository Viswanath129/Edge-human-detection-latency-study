import sys
import os
import torch

# Add the current directory to sys.path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_tests():
    model_path = "yolov8n.pt"
    resolution = 640
    # Test FP32 and FP16
    precisions = [False, True]

    print(f"Starting precision tests with {model_path} at {resolution}x{resolution}...")

    if not torch.cuda.is_available():
        print("CUDA not available. FP16 tests will fall back to FP32 or may be skipped.")

    for half in precisions:
        prec_name = "FP16" if half else "FP32"
        print(f"Testing precision: {prec_name}")

        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=resolution, half=half)

        # If we requested half but didn't get it, and we already tested FP32, skip saving
        if half and not actual_half:
            print(f"FP16 requested but not available on this hardware. Skipping result update.")
            continue

        save_summary(
            resolution=f"{resolution}x{resolution}",
            model_name="YOLOv8n",
            precision=prec_name,
            fps=fps,
            latency=avg_latency,
            observation=f"Standard {prec_name} inference"
        )
        print(f"Precision {prec_name} complete: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_precision_tests()
