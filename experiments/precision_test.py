from utils import benchmark_model, save_summary

def run_precision_test():
    model_path = "yolov8n.pt"
    imgsz = 640

    # FP32 Test
    print("Running FP32 Benchmark...")
    fps_32, latency_32 = benchmark_model(model_path, imgsz=imgsz, half=False)
    save_summary(f"{imgsz}x{imgsz}", "YOLOv8n", "FP32", fps_32, latency_32, "Standard precision")

    # FP16 Test
    print("Running FP16 Benchmark...")
    fps_16, latency_16 = benchmark_model(model_path, imgsz=imgsz, half=True)
    save_summary(f"{imgsz}x{imgsz}", "YOLOv8n", "FP16", fps_16, latency_16, "Half precision (optimized for GPU)")

if __name__ == "__main__":
    run_precision_test()
