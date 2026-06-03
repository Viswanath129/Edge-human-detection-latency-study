from utils import benchmark_model, save_summary

def run_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_path in models:
        model_name = model_path.replace(".pt", "")
        print(f"Benchmarking model size: {model_name}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Lightweight nano model" if "nano" in model_name or "8n" in model_name else "Standard small model"
        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_test()
