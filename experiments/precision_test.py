from utils import benchmark_model, save_summary

def run_test():
    precisions = [False, True] # False = FP32, True = FP16 (half)
    res = 640
    model_path = "yolov8n.pt"

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {precision_name}")
        fps, latency = benchmark_model(model_path, imgsz=res, half=half)

        observation = "Standard precision" if not half else "Half precision (optimized for GPU/NPU)"
        save_summary(
            resolution=f"{res}x{res}",
            model_name="yolov8n",
            precision=precision_name,
            fps=fps,
            latency=latency,
            observation=observation
        )

if __name__ == "__main__":
    run_test()
