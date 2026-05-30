from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Running Resolution Test: 640x640 vs 416x416")

    # 640x640 Benchmark
    fps_640, lat_640 = benchmark_model("yolov8n.pt", imgsz=640, half=False)
    save_summary("640x640", "Nano", "FP32", fps_640, lat_640, "Higher detection quality")

    # 416x416 Benchmark
    fps_416, lat_416 = benchmark_model("yolov8n.pt", imgsz=416, half=False)
    save_summary("416x416", "Nano", "FP32", fps_416, lat_416, "Faster Inference")

if __name__ == "__main__":
    run_resolution_test()
