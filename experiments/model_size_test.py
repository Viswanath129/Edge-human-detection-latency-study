from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Running Model Size Comparison Test...")

    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        print(f"Benchmarking model: {model_name}")
        avg_latency, fps, _ = benchmark_model(model_name=model_name, imgsz=res)

        observation = "Nano (Fastest)" if "yolov8n" in model_name else "Small (Balanced)"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, observation)

if __name__ == "__main__":
    run_model_size_test()
