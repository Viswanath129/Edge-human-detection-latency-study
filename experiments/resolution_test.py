from utils import benchmark_model, save_summary

def run_resolution_test():
    print("Running Resolution Test...")
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model_name, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_resolution_test()
