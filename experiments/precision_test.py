from utils import benchmark_model, save_summary

def run_precision_test():
    precisions = ["FP32", "FP16"]
    model_name = "yolov8n.pt"
    resolution = 640

    for prec in precisions:
        fps, latency = benchmark_model(model_name, resolution, precision=prec)

        observation = "Standard precision" if prec == "FP32" else "Optimized for CUDA/NPU; may be slower on CPU"

        save_summary(
            resolution=resolution,
            model="yolov8n",
            precision=prec,
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_precision_test()
