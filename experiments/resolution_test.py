from ultralytics import YOLO
from utils import benchmark_model, save_summary
import os

def run_resolution_test():
    model = YOLO("yolov8n.pt")
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_resolution_test()
