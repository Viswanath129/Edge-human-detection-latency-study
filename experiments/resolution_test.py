from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        print(f"\n--- Testing Resolution: {res}x{res} ---")
        fps, latency = benchmark_model(model_name, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "yolov8n", "FP32", fps, latency, observation)

        print(f"Results for {res}x{res}: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
