from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_path in models:
        model_name = "YOLOv8n" if "n" in model_path else "YOLOv8s"
        print(f"Benchmarking model: {model_name}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Lightweight / Edge optimized" if "n" in model_path else "Improved accuracy / Higher compute"
        save_summary(f"{res}x{res}", model_name, "FP32", fps, latency, observation)
        print(f"Results: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    main()
