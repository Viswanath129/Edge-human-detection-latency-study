from ultralytics import YOLO
from utils import benchmark_model, save_summary
import torch

def run_precision_test():
    model = YOLO("yolov8n.pt")

    # FP32
    print("Benchmarking FP32 precision")
    avg_latency, fps, _ = benchmark_model(model, imgsz=640, half=False)
    save_summary("640x640", "yolov8n", "FP32", fps, avg_latency, "Standard precision")

    # FP16
    print("Benchmarking FP16 precision")
    avg_latency, fps, actual_half = benchmark_model(model, imgsz=640, half=True)
    if actual_half:
        save_summary("640x640", "yolov8n", "FP16", fps, avg_latency, "Half precision (CUDA)")
    else:
        print("Skipping FP16 summary update as it was not supported.")

if __name__ == "__main__":
    run_precision_test()
