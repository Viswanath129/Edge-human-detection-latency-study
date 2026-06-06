import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Running Model Size (Nano vs Small) Test...")

    # Test YOLOv8n (Nano)
    fps_n, lat_n = benchmark_model("yolov8n.pt", imgsz=640)
    save_summary("640x640", "YOLOv8n", "FP32", fps_n, lat_n, "Lightweight nano model")

    # Test YOLOv8s (Small)
    fps_s, lat_s = benchmark_model("yolov8s.pt", imgsz=640)
    save_summary("640x640", "YOLOv8s", "FP32", fps_s, lat_s, "Medium small model")

if __name__ == "__main__":
    run_model_size_test()
