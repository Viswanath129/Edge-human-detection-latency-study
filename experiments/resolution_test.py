import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        precision = "FP16" if actual_half else "FP32"

        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    main()
