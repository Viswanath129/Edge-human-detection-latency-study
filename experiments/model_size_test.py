from utils import benchmark_model, save_summary

def main():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]
    precision = "FP32"

    for model_path in models:
        model_name = model_path.split(".")[0]
        print(f"Benchmarking model size: {model_name}")
        fps, latency = benchmark_model(model_path, res)

        observation = "Lightweight nano model" if "nano" in model_name or "yolov8n" in model_name else "Improved accuracy, higher compute"
        save_summary(f"{res}x{res}", model_name, precision, fps, latency, observation)

if __name__ == "__main__":
    main()
