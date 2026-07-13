from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    for m in models:
        print(f"\n--- Testing Model: {m} ---")
        avg_latency, fps, _ = benchmark_model(model_name=m, imgsz=640, half=False)

        obs = "Baseline Nano model" if "n" in m else "Improved accuracy, higher latency"
        save_summary("640x640", m, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_model_size_test()
