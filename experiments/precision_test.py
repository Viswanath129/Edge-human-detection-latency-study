from utils import benchmark_model, save_summary

def run_precision_test():
    print("Starting Precision Benchmark...")

    model_path = "yolov8n.pt"
    resolution = 640
    precisions = [False, True]  # False = FP32, True = FP16

    for half in precisions:
        p_name = "FP16" if half else "FP32"
        print(f"Testing precision: {p_name}...")
        fps, latency = benchmark_model(model_path, imgsz=resolution, half=half)

        observation = "Standard precision" if not half else "Optimized for GPU/NPU"
        save_summary(f"{resolution}x{resolution}", "yolov8n", p_name, fps, latency, observation)

        print(f"Results for {p_name}: {fps} FPS, {latency} ms")

if __name__ == "__main__":
    run_precision_test()
