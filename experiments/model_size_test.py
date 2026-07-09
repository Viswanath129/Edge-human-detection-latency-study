import os
from utils import benchmark_model, save_summary

def main():
    models = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt"
    }
    res = 640

    for model_name, model_path in models.items():
        print(f"Benchmarking Model: {model_name}")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        observation = "Ultra-lightweight" if "nano" in model_name or "v8n" in model_name else "Balanced performance"
        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

        print(f"Results for {model_name}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()
