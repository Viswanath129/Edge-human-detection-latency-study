import sys
import os

# Ensure local imports work if run as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_tests():
    res = 640
    model_name = "yolov8n.pt"

    # FP32
    print("--- Benchmarking Precision: FP32 ---")
    latency, fps, is_half = benchmark_model(model_name, res, half=False)
    save_summary(res, "yolov8n", "FP32", fps, latency, "Standard precision")

    # FP16
    print("--- Benchmarking Precision: FP16 ---")
    latency, fps, is_half = benchmark_model(model_name, res, half=True)
    if is_half:
        save_summary(res, "yolov8n", "FP16", fps, latency, "Accelerated precision")
    else:
        print("FP16 not supported/available on this hardware, skipping summary update.")

if __name__ == "__main__":
    run_precision_tests()
