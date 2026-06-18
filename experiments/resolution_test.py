import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        res_str = f"{res}x{res}"
        obs = "Higher detection quality" if res == 640 else "Faster Inference"

        save_summary(res_str, "yolov8n", precision, fps, avg_latency, obs)

if __name__ == "__main__":
    main()
