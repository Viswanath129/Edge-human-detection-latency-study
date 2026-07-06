import os
import sys
import torch

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_tests():
    model_name = "yolov8n.pt"
    res = 640

    precisions = [False, True] # False=FP32, True=FP16

    for half in precisions:
        precision_label = "FP16" if half else "FP32"
        print(f"\n--- Benchmarking Precision: {precision_label} ---")

        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=half)

        if half and not actual_half:
            print("Skipping FP16 results as it was not supported/used.")
            continue

        observation = "Standard precision" if not actual_half else "Half precision optimization"
        save_summary(f"{res}x{res}", "yolov8n", precision_label, fps, avg_latency, observation)

        print(f"Result: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_precision_tests()
