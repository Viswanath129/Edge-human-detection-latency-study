import os
import sys

# Ensure local imports work when running from the script directory or root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        observation = "Standard resolution" if res == 640 else "High-speed optimized"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, observation)

        print(f"Resolution {res}x{res} - Latency: {avg_latency:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
