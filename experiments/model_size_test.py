from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        # Extract model variant name (e.g., yolov8n)
        variant = model_name.split('.')[0]
        print(f"Benchmarking model size: {variant}")
        fps, latency = benchmark_model(model_name, imgsz=res)

        observation = "Ultra-lightweight" if "n" in variant else "Balanced performance"
        save_summary(f"{res}x{res}", variant, "FP32", fps, latency, observation)

if __name__ == "__main__":
    run_model_size_test()
