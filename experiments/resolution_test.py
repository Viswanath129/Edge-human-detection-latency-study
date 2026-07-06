import os
import sys

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_tests():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Benchmarking Resolution: {res}x{res} ---")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, observation)

        print(f"Result: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_tests()
