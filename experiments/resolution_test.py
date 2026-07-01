import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        obs = "Higher quality" if res == 640 else "Faster inference"

        save_summary(f"{res}x{res}", "YOLOv8n", precision, fps, avg_latency, obs)

if __name__ == "__main__":
    # Force synthetic for headless environments if not specified
    if 'FORCE_SYNTHETIC' not in os.environ:
        os.environ['FORCE_SYNTHETIC'] = 'true'
    run_resolution_test()
