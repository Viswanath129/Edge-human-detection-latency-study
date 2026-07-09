import os
from utils import benchmark_model, save_summary

def main():
    model_path = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, _ = benchmark_model(model_path, imgsz=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

        print(f"Results for {res}x{res}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()
