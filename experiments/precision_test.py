import torch
from utils import benchmark_model, save_summary

def run_precision_test():
    model_name = "yolov8n.pt"
    res = 640

    # FP32
    avg_latency_32, fps_32, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "yolov8n", "FP32", fps_32, avg_latency_32, "Standard precision")

    # FP16
    if torch.cuda.is_available():
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", "yolov8n", "FP16", fps_16, avg_latency_16, "Half precision (CUDA)")
    else:
        print("Skipping FP16 test as CUDA is not available.")

if __name__ == "__main__":
    run_precision_test()
