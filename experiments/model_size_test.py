import sys
import os

# Ensure local imports work if run as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_tests():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_name in models:
        clean_name = model_name.split('.')[0]
        print(f"--- Benchmarking Model Size: {clean_name} ---")
        latency, fps, is_half = benchmark_model(model_name, res)

        precision = "FP16" if is_half else "FP32"
        obs = "Ultra-lightweight" if "n" in clean_name else "Balanced performance"

        save_summary(res, clean_name, precision, fps, latency, obs)

if __name__ == "__main__":
    run_model_size_tests()
