import torch
from utils import benchmark_model, save_summary

def main():
    model_name = "yolov8n.pt"
    res = 640

    # FP32
    avg_latency, fps, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, "Standard precision")

    # FP16
    if torch.cuda.is_available():
        avg_latency_h, fps_h, actual_half = benchmark_model(model_name, imgsz=res, half=True)
        if actual_half:
            save_summary(f"{res}x{res}", "yolov8n", "FP16", fps_h, avg_latency_h, "Half precision (GPU accelerated)")
    else:
        print("Skipping FP16 update in summary.csv as CUDA is not available.")

if __name__ == "__main__":
    main()
