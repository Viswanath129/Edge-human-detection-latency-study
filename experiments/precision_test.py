import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    print("Running Precision (FP32 vs FP16) Test...")

    # Test FP32 (Standard)
    fps_32, lat_32 = benchmark_model("yolov8n.pt", imgsz=640, half=False)
    save_summary("640x640", "YOLOv8n", "FP32", fps_32, lat_32, "Baseline precision")

    # Test FP16 (Half)
    fps_16, lat_16 = benchmark_model("yolov8n.pt", imgsz=640, half=True)
    save_summary("640x640", "YOLOv8n", "FP16", fps_16, lat_16, "Reduced precision for speedup")

if __name__ == "__main__":
    run_precision_test()
