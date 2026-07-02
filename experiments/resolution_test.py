import sys
import os

# Add the experiments directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        latency, fps, half_used = benchmark_model(model_name, imgsz=res)
        precision = "FP16" if half_used else "FP32"

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", model_name, precision, fps, latency, obs)
        print(f"Results: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
