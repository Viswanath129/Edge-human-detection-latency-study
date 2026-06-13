import os
import sys

# Add the experiments directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_latency, fps, is_half = benchmark_model(model_name, imgsz=res, half=False)

        precision = "FP16" if is_half else "FP32"
        # Unique observation for 640 to avoid being overwritten by model_size_test
        obs = "Resolution baseline: 640x640" if res == 640 else "Faster Inference"

        save_summary(res, fps, avg_latency, "yolov8n", precision, obs)

if __name__ == "__main__":
    run_resolution_test()
