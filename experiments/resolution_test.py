from utils import benchmark_model, save_summary

def run_resolution_test():
    model_path = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Running Benchmark at {res}x{res}...")
        fps, latency = benchmark_model(model_path, imgsz=res)
        save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps, latency, f"Resolution study ({res}px)")

if __name__ == "__main__":
    run_resolution_test()
