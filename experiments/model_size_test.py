import os
import sys

# Add the experiments directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_name in models:
        short_name = model_name.split('.')[0]
        print(f"\n--- Testing Model: {short_name} ---")

        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if is_half else "FP32"
        obs = f"Model variant: {short_name}"

        save_summary(res, fps, avg_latency, short_name, precision, obs)

if __name__ == "__main__":
    run_model_size_test()
