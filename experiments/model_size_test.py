import os
import sys

# Ensure experiments directory is in path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"Testing model size: {model_name}...")
        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if actual_half else "FP32"
        short_name = model_name.split('.')[0]
        observation = "Nano model" if "yolov8n" in model_name else "Small model"

        save_summary(f"{res}x{res}", fps, avg_latency, observation, short_name, precision)

if __name__ == "__main__":
    run_model_size_test()
