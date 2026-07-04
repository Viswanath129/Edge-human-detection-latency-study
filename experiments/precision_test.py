import os
import sys
import torch

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_tests():
    precisions = [False, True] # False = FP32, True = FP16
    model_name = "yolov8n.pt"
    resolution = 640

    for half in precisions:
        precision_label = "FP16" if half else "FP32"

        # FP16 often requires CUDA for actual speedup, benchmark_model handles the 'half' parameter
        print(f"\n--- Testing Precision: {precision_label} ---")

        avg_latency, fps, actual_half = benchmark_model(model_name, resolution, half=half)

        # If we requested half but didn't get it (e.g. no CUDA), note it
        observation = "Standard precision"
        if half:
            if actual_half:
                observation = "Hardware accelerated FP16"
            else:
                observation = "FP16 requested but not hardware supported (falling back)"
                # Don't update if it's just a fallback to FP32 to avoid duplicate entries under different labels
                # unless you want to show it's same. Actually let's just note it.
                if not torch.cuda.is_available():
                    print("Skipping FP16 summary update as CUDA is not available.")
                    continue

        save_summary(f"{resolution}x{resolution}", "yolov8n", precision_label, fps, avg_latency, observation)
        print(f"Results for {precision_label}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_precision_tests()
