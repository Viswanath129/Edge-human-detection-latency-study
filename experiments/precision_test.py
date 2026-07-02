import sys
import os

# Add the experiments directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_precision_test():
    res = 640
    model_name = "yolov8n.pt"

    # FP32
    print("Benchmarking precision: FP32")
    latency, fps, _ = benchmark_model(model_name, imgsz=res, half=False)
    save_summary(f"{res}x{res}", model_name, "FP32", fps, latency, "Standard precision")
    print(f"FP32 Results: {fps:.2f} FPS, {latency:.2f} ms")

    # FP16
    print("Benchmarking precision: FP16")
    latency, fps, half_used = benchmark_model(model_name, imgsz=res, half=True)
    if half_used:
        save_summary(f"{res}x{res}", model_name, "FP16", fps, latency, "Hardware accelerated")
        print(f"FP16 Results: {fps:.2f} FPS, {latency:.2f} ms")
    else:
        print("Skipping FP16 results recording as CUDA was not available.")

if __name__ == "__main__":
    run_precision_test()
