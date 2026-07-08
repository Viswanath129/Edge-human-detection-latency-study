import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"\n--- Benchmarking Model: {model_name} ---")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        observation = "Lightweight nano model" if "v8n" in model_name else "Standard small model"
        save_summary(res, model_name, "FP32", fps, avg_latency, observation)

        print(f"Avg Latency: {avg_latency:.2f} ms")
        print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
