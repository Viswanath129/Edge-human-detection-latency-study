from utils import benchmark_model, save_summary
import torch

def run_precision_test():
    model_path = "yolov8n.pt"
    model_name = "yolov8n"
    resolution = "640x640"
    res_val = 640

    # FP32
    print("Benchmarking precision: FP32")
    avg_lat_32, fps_32, _ = benchmark_model(model_path, resolution=res_val, half=False)
    save_summary(
        resolution=resolution,
        model_name=model_name,
        precision="FP32",
        fps=fps_32,
        latency=avg_lat_32,
        observation="Standard precision"
    )

    # FP16
    print("Benchmarking precision: FP16")
    avg_lat_16, fps_16, is_half = benchmark_model(model_path, resolution=res_val, half=True)

    if is_half:
        save_summary(
            resolution=resolution,
            model_name=model_name,
            precision="FP16",
            fps=fps_16,
            latency=avg_lat_16,
            observation="Half precision (requires CUDA for speedup)"
        )
    else:
        print("Skipping FP16 summary update as it fell back to FP32.")

if __name__ == "__main__":
    run_precision_test()
