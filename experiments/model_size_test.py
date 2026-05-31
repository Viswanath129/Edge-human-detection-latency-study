from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for m in models:
        model_label = m.split('.')[0]
        print(f"\n--- Testing Model: {model_label} ---")
        fps, latency = benchmark_model(m, imgsz=res)

        observation = "Lightweight nano model" if "n" in model_label else "Balanced small model"
        save_summary(f"{res}x{res}", model_label, "FP32", fps, latency, observation)

        print(f"Results for {model_label}: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
