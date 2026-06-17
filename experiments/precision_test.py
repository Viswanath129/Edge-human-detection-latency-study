from ultralytics import YOLO
from utils import benchmark_model, save_summary
import torch

def run_precision_test():
    model_name = "yolov8n"
    model = YOLO(f"{model_name}.pt")

    # Test FP32
    print(f"Benchmarking {model_name} at FP32 precision...")
    avg_latency_32, fps_32, _ = benchmark_model(model, imgsz=640, half=False)
    save_summary("640x640", model_name, "FP32", fps_32, avg_latency_32, "Standard precision")

    # Test FP16
    print(f"Benchmarking {model_name} at FP16 precision...")
    # Note: FP16 usually requires GPU (CUDA) for speedup
    avg_latency_16, fps_16, actual_half = benchmark_model(model, imgsz=640, half=True)

    obs = "Accelerated precision" if actual_half and torch.cuda.is_available() else "FP16 (Software fallback if no GPU)"
    save_summary("640x640", model_name, "FP16", fps_16, avg_latency_16, obs)

    print(f"FP32: {fps_32:.2f} FPS, {avg_latency_32:.2f} ms")
    print(f"FP16: {fps_16:.2f} FPS, {avg_latency_16:.2f} ms")

if __name__ == "__main__":
    run_precision_test()
