import os
import sys

# Add the experiments directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    res = 640
    model_name = "yolov8n.pt"
    precision_modes = [False, True] # [FP32, FP16]

    for half in precision_modes:
        mode_str = "FP16" if half else "FP32"
        print(f"\n--- Testing Precision: {mode_str} ---")

        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=half)

        actual_precision = "FP16" if is_half else "FP32"
        obs = f"Precision test: {actual_precision}"

        save_summary(res, fps, avg_latency, "yolov8n", actual_precision, obs)

if __name__ == "__main__":
    run_precision_test()
