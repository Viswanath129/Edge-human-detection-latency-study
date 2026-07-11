import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_test():
    model_name = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        observation = "Standard resolution" if res == 640 else "Downscaled for speed"
        save_summary(res, model_name.replace(".pt", ""), "FP32", fps, avg_latency, observation)

        print(f"Resolution: {res}x{res}, FPS: {fps:.2f}, Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
