import os
import sys
from ultralytics import YOLO

# Add parent directory to path to allow importing utils if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def main():
    model = YOLO("yolov8n.pt")

    precisions = [False, True] # False = FP32, True = FP16

    for half in precisions:
        prec_name = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {prec_name}")
        avg_latency, fps, actual_half = benchmark_model(model, imgsz=640, half=half)

        actual_prec_name = "FP16" if actual_half else "FP32"
        obs = "Standard precision" if not actual_half else "Reduced precision for speed"

        save_summary(
            resolution="640x640",
            model_name="YOLOv8n",
            precision=actual_prec_name,
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    main()
