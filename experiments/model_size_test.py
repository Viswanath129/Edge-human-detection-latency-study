from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_name in models:
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        size_label = "nano" if "yolov8n" in model_name else "small"
        observation = f"Standard {size_label} model performance"
        save_summary(f"{res}x{res}", model_name.replace(".pt", ""), "FP32", fps, avg_latency, observation)

if __name__ == "__main__":
    run_model_size_test()
