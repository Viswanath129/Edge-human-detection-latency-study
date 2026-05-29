from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        fps, latency = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(f"{res}x{res}", "YOLOv8n", "FP32", fps, latency, observation)
        print(f"Results: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    main()
