import torch
from utils import benchmark_model, save_summary

def run_precision_test():
    # Only test FP32 if CUDA is not available, as FP16 on CPU is not representative
    precisions = [False]
    if torch.cuda.is_available():
        precisions.append(True)
    else:
        print("CUDA not available. Skipping FP16 (Half Precision) benchmark.")

    for half in precisions:
        label = "FP16" if half else "FP32"
        print(f"\n--- Testing Precision: {label} ---")
        avg_latency, fps, actual_half = benchmark_model(model_name="yolov8n.pt", imgsz=640, half=half)

        # If it fell back to FP32 despite requesting half, don't record as FP16
        if half and not actual_half:
            continue

        obs = "Standard precision" if not half else "Hardware-accelerated inference"
        save_summary("640x640", "yolov8n.pt", label, fps, avg_latency, obs)

if __name__ == "__main__":
    run_precision_test()
