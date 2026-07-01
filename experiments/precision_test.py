import os
import sys
import torch

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    res = 640
    model_path = "yolov8n.pt"

    # Test FP32
    print("Testing precision: FP32")
    avg_latency_32, fps_32, _ = benchmark_model(model_path, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps_32, avg_latency_32, "Standard precision")

    # Test FP16 if CUDA available
    if torch.cuda.is_available():
        print("Testing precision: FP16")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_path, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", "YOLOv8n", "FP16", fps_16, avg_latency_16, "Reduced precision, hardware accelerated")
    else:
        print("FP16 test skipped (CUDA not available)")

if __name__ == "__main__":
    if 'FORCE_SYNTHETIC' not in os.environ:
        os.environ['FORCE_SYNTHETIC'] = 'true'
    run_precision_test()
