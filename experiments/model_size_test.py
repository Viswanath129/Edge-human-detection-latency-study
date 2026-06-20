import sys
import os
from ultralytics import YOLO

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    names = ["YOLOv8n", "YOLOv8s"]

    res = 640

    for model_path, model_name in zip(models, names):
        print(f"Benchmarking model size: {model_name}...")
        model = YOLO(model_path)
        avg_latency, fps, _ = benchmark_model(model, imgsz=res)

        obs = "Nano model (lightweight)" if model_name == "YOLOv8n" else "Small model (more accurate)"
        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )
        print(f"Model: {model_name} | Avg Latency: {avg_latency:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    run_test()
