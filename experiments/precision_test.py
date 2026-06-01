import os
import sys
import torch
from ultralytics import YOLO

# Add parent directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    model_path = "yolov8n.pt"
    res = 640

    # FP32
    print("Benchmarking Precision: FP32")
    model = YOLO(model_path)
    fps_32, lat_32 = benchmark_model(model, input_size=res)
    save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps_32, lat_32, "Standard precision")
    print(f"FP32: {fps_32:.2f} FPS, {lat_32:.2f} ms")

    # FP16
    print("Benchmarking Precision: FP16")
    # half=True in model call
    fps_16, lat_16 = benchmark_model(model, input_size=res, half=True)

    obs = "FP16 optimized"
    if not torch.cuda.is_available():
        obs += " (CPU fallback - may be slower)"

    save_summary(f"{res}x{res}", "YOLOv8n", "FP16", fps_16, lat_16, obs)
    print(f"FP16: {fps_16:.2f} FPS, {lat_16:.2f} ms")

if __name__ == "__main__":
    run_precision_test()
