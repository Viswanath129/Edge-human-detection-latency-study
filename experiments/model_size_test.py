import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    res = 640
    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt"
    }

    for name, path in models.items():
        print(f"Testing model size: {name}")
        avg_latency, fps, actual_half = benchmark_model(path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        obs = "Nano - Optimized for speed" if "n" in path else "Small - Better accuracy"

        save_summary(f"{res}x{res}", name, precision, fps, avg_latency, obs)

if __name__ == "__main__":
    if 'FORCE_SYNTHETIC' not in os.environ:
        os.environ['FORCE_SYNTHETIC'] = 'true'
    run_model_size_test()
