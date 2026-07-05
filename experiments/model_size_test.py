import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"Benchmarking model size: {model_name}")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        short_name = model_name.replace(".pt", "")
        observation = "Highly efficient nano model" if "n" in model_name else "Balanced small model"
        save_summary(f"{res}x{res}", short_name, "FP32", fps, avg_latency, observation)

        print(f"Model {short_name} - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
