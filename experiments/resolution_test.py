from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        avg_latency, fps, _ = benchmark_model(model_name="yolov8n.pt", imgsz=res, half=False)

        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n.pt", "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_resolution_test()
