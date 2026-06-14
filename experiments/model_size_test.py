import os
import sys

# Add the current directory to sys.path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    # Compare Nano vs Small
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"--- Benchmarking Model: {model_name} ---")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        obs = "Ultra-lightweight" if "n" in model_name else "Better accuracy balance"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, obs)

        print(f"Results for {model_name}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
