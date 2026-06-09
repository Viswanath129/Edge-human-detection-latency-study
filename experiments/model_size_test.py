from utils import benchmark_model, save_summary

def run_model_size_test():
    imgsz = 640

    # YOLOv8n (Nano)
    print("Running YOLOv8n Benchmark...")
    fps_n, latency_n = benchmark_model("yolov8n.pt", imgsz=imgsz)
    save_summary(f"{imgsz}x{imgsz}", "YOLOv8n", "FP32", fps_n, latency_n, "Nano model (edge optimized)")

    # YOLOv8s (Small)
    print("Running YOLOv8s Benchmark...")
    fps_s, latency_s = benchmark_model("yolov8s.pt", imgsz=imgsz)
    save_summary(f"{imgsz}x{imgsz}", "YOLOv8s", "FP32", fps_s, latency_s, "Small model (higher accuracy)")

if __name__ == "__main__":
    run_model_size_test()
