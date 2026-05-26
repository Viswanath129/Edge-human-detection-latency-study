from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [640, 416]
    model_name = "yolov8n.pt"

    for res in resolutions:
        fps, latency = benchmark_model(model_name, res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"

        save_summary(
            resolution=res,
            model="yolov8n",
            precision="FP32",
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_resolution_test()
