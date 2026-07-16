from utils import benchmark_model, save_summary
import torch

def run_precision_test():
    print("Running Precision (FP32 vs FP16) Test...")

    model_name = "yolov8n.pt"
    res = 640

    # FP32
    print("Benchmarking FP32 precision...")
    avg_latency_32, fps_32, _ = benchmark_model(model_name=model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", model_name, "FP32", fps_32, avg_latency_32, "Standard precision")

    # FP16
    if torch.cuda.is_available():
        print("Benchmarking FP16 precision...")
        avg_latency_16, fps_16, actual_half = benchmark_model(model_name=model_name, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", model_name, "FP16", fps_16, avg_latency_16, "Reduced precision, faster on GPU")
    else:
        print("Skipping FP16 test: CUDA not available.")

if __name__ == "__main__":
    run_precision_test()
