from utils import benchmark_model, save_summary

def main():
    res = 640
    model_path = "yolov8n.pt"
    model_name = "yolov8n"
    precisions = ["FP32", "FP16"]

    for prec in precisions:
        print(f"Benchmarking precision: {prec}")
        half = (prec == "FP16")
        fps, latency = benchmark_model(model_path, res, half=half)

        observation = "Standard precision" if prec == "FP32" else "Optimized for hardware acceleration"
        save_summary(f"{res}x{res}", model_name, prec, fps, latency, observation)

if __name__ == "__main__":
    main()
