import sys
import os

# Add the experiments directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_name in models:
        print(f"Benchmarking model size: {model_name}")
        latency, fps, half_used = benchmark_model(model_name, imgsz=res)
        precision = "FP16" if half_used else "FP32"

        obs = "Ultra-lightweight" if "n" in model_name else "Balanced performance"
        save_summary(f"{res}x{res}", model_name, precision, fps, latency, obs)
        print(f"Results: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
