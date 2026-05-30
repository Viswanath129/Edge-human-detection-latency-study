from utils import benchmark_model, save_summary

def run_precision_test():
    print("Running Precision Test: FP32 vs FP16")

    # FP32 Benchmark
    fps_32, lat_32 = benchmark_model("yolov8n.pt", imgsz=640, half=False)
    save_summary("640x640", "Nano", "FP32", fps_32, lat_32, "Standard precision")

    # FP16 Benchmark
    fps_16, lat_16 = benchmark_model("yolov8n.pt", imgsz=640, half=True)
    save_summary("640x640", "Nano", "FP16", fps_16, lat_16, "Half precision (optimized for GPU/NPU)")

if __name__ == "__main__":
    run_precision_test()
