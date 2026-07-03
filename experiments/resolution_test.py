import os
import sys

# Ensure experiments directory is in path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}...")
        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if actual_half else "FP32"
        observation = "Standard resolution" if res == 640 else "Lower resolution for speed"

        save_summary(f"{res}x{res}", fps, avg_latency, observation, "yolov8n", precision)

if __name__ == "__main__":
    run_resolution_test()
