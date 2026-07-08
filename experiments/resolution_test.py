import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Benchmarking Resolution: {res}x{res} ---")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(res, model_name, "FP32", fps, avg_latency, observation)

        print(f"Avg Latency: {avg_latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
