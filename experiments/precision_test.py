import torch
from utils import benchmark_model, save_summary

def run_precision_test():
    print("Running Precision Test...")
    model_name = "yolov8n.pt"
    res = 640

    # FP32
    print("Testing FP32...")
    avg_latency, fps, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, "Standard precision")

    # FP16 (only if CUDA is available)
    if torch.cuda.is_available():
        print("Testing FP16...")
        avg_latency, fps, actual_half = benchmark_model(model_name, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", model_name, "FP16", fps, avg_latency, "Hardware accelerated")
    else:
        print("Skipping FP16 test: CUDA not available")

if __name__ == "__main__":
    run_precision_test()
