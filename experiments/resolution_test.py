import os
import sys

# Add the current directory to sys.path to allow local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"--- Benchmarking Resolution: {res}x{res} ---")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, obs)

        print(f"Results for {res}x{res}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
