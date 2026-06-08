import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_tests():
    model_path = "yolov8n.pt"
    imgsz = 640
    precisions = [False, True] # False = FP32, True = FP16

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"\n--- Testing Precision: {precision_name} ---")
        avg_fps, avg_latency = benchmark_model(model_path, imgsz=imgsz, half=half)

        observation = "Standard precision" if not half else "Reduced precision for acceleration"
        save_summary(
            resolution=f"{imgsz}x{imgsz}",
            model_name="YOLOv8n",
            precision=precision_name,
            avg_fps=avg_fps,
            avg_latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_precision_tests()
