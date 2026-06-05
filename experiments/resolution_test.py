from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Starting Resolution Benchmark...")

    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}...")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, latency, observation)

        print(f"Results for {res}x{res}: {fps} FPS, {latency} ms")

if __name__ == "__main__":
    run_resolution_test()
