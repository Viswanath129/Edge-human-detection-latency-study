from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n", "yolov8s"]
    resolution = 640

    for m in models:
        model_file = f"{m}.pt"
        fps, latency = benchmark_model(model_file, resolution)

        observation = "Nano: ultra-fast for edge" if m == "yolov8n" else "Small: better accuracy, higher latency"

        save_summary(
            resolution=resolution,
            model=m,
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_model_size_test()
