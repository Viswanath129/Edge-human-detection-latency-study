import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Running Resolution Comparison Test...")

    # Test 640x640
    fps_640, lat_640 = benchmark_model("yolov8n.pt", imgsz=640)
    save_summary("640x640", "YOLOv8n", "FP32", fps_640, lat_640, "Higher detection quality")

    # Test 416x416
    fps_416, lat_416 = benchmark_model("yolov8n.pt", imgsz=416)
    save_summary("416x416", "YOLOv8n", "FP32", fps_416, lat_416, "Faster Inference")

if __name__ == "__main__":
    run_resolution_test()
