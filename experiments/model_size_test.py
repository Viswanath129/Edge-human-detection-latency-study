import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_name in models:
        name_short = model_name.split('.')[0]
        print(f"Benchmarking Model: {name_short}")
        avg_latency, fps, half = benchmark_model(model_name, imgsz=res)

        observation = "Ultra-lightweight" if "yolov8n" in model_name else "Balanced accuracy/speed"
        save_summary(f"{res}x{res}", name_short, "FP32", fps, avg_latency, observation)

if __name__ == "__main__":
    run_model_size_test()
