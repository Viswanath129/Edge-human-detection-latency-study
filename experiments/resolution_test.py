import sys
import os
from ultralytics import YOLO

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    # Load model
    model = YOLO("yolov8n.pt")

    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}...")
        avg_latency, fps, _ = benchmark_model(model, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="YOLOv8n",
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )
        print(f"Res: {res} | Avg Latency: {avg_latency:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    run_test()
