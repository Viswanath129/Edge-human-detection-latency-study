import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    # Comparing nano and small models at 640x640
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_path in models:
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        res_str = f"{res}x{res}"
        model_name = model_path.split('.')[0]
        obs = "Baseline Nano" if "n" in model_path else "Higher capacity Small"

        save_summary(res_str, model_name, precision, fps, avg_latency, obs)

if __name__ == "__main__":
    main()
