import os
import sys

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_tests():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        print(f"\n--- Benchmarking Model: {model_name} ---")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        observation = "Ultra-lightweight" if "nano" in model_name or "yolov8n" in model_name else "Improved accuracy, higher latency"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, observation)

        print(f"Result: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_tests()
