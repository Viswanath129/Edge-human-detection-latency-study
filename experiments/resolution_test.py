from utils import benchmark_model, save_summary

def main():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"
    model_name = "yolov8n"
    precision = "FP32"

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        fps, latency = benchmark_model(model_path, res)

        observation = "Higher detection quality" if res == 640 else "Faster inference"
        save_summary(f"{res}x{res}", model_name, precision, fps, latency, observation)

if __name__ == "__main__":
    main()
