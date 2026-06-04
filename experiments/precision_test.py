import os
import sys
from ultralytics import YOLO

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    print("Starting Precision Test (FP32 vs FP16)...")
    model = YOLO("yolov8n.pt")

    # FP16 might be slower on CPU but faster on GPU/NPU
    precisions = [False, True] # False = FP32, True = FP16 (half)

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {precision_name}")

        fps, latency = benchmark_model(model, imgsz=640, half=half)

        observation = "Standard precision" if not half else "Half precision (CPU non-optimized)"
        save_summary(
            resolution="640x640",
            model_name="yolov8n",
            precision=precision_name,
            fps=fps,
            latency=latency,
            observation=observation
        )
        print(f"Precision {precision_name} - FPS: {fps:.2f}, Latency: {latency:.2f} ms")

if __name__ == "__main__":
    run_precision_test()
