import sys
import os
import torch

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    model_name = "yolov8n.pt"
    res = 640
    precisions = [False, True] # False=FP32, True=FP16

    for half in precisions:
        precision_label = "FP16" if half else "FP32"
        print(f"\n--- Benchmarking Precision: {precision_label} ---")

        if half and not torch.cuda.is_available():
            print("Skipping FP16 test: CUDA not available. FP16 on CPU is usually slower and not representative.")
            continue

        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=half)

        # If requested half but didn't get it (due to no cuda), label as FP32 anyway?
        # benchmark_model returns actual_half
        final_precision = "FP16" if actual_half else "FP32"

        observation = "Standard precision" if final_precision == "FP32" else "Half precision (CUDA accelerated)"
        save_summary(res, model_name, final_precision, fps, avg_latency, observation)

        print(f"Avg Latency: {avg_latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
