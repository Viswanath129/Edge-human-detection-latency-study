import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    precisions = [False, True]  # False = FP32, True = FP16
    model_path = "yolov8n.pt"
    resolution = 640

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Testing precision: {precision_name}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=resolution, half=half)

        # If hardware doesn't support FP16, it might have fallen back
        actual_precision = "FP16" if actual_half else "FP32"

        observation = "Standard precision" if actual_precision == "FP32" else "Hardware accelerated inference"

        save_summary(
            resolution=f"{resolution}x{resolution}",
            model_name="yolov8n",
            precision=actual_precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    main()
