from utils import benchmark_model, save_summary

def run_test():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Standard resolution" if res == 640 else "High-speed optimized"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_test()
