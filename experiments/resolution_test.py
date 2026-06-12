import sys
import os

# Ensure local imports work if run as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_tests():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"--- Benchmarking Resolution: {res}x{res} ---")
        latency, fps, is_half = benchmark_model(model_name, res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        precision = "FP16" if is_half else "FP32"

        save_summary(res, "yolov8n", precision, fps, latency, obs)

if __name__ == "__main__":
    run_resolution_tests()
