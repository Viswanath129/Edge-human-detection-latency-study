from utils import benchmark_model, save_summary

def run_test():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
