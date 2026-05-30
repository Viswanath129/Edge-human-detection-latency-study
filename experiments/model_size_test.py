from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Running Model Size Test: Nano vs Small")

    # Nano (yolov8n) Benchmark
    fps_n, lat_n = benchmark_model("yolov8n.pt", imgsz=640, half=False)
    save_summary("640x640", "Nano", "FP32", fps_n, lat_n, "Smallest variant")

    # Small (yolov8s) Benchmark
    fps_s, lat_s = benchmark_model("yolov8s.pt", imgsz=640, half=False)
    save_summary("640x640", "Small", "FP32", fps_s, lat_s, "Medium performance/accuracy balance")

if __name__ == "__main__":
    run_model_size_test()
