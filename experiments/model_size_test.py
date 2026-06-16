from utils import benchmark_model, save_summary

def run_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for m in models:
        model_name = m.replace(".pt", "")
        print(f"Testing model size: {model_name}")
        avg_latency, fps, _ = benchmark_model(m, imgsz=res)

        obs = "Ultra-lightweight" if "nano" in model_name or "8n" in model_name else "Improved accuracy, higher latency"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
