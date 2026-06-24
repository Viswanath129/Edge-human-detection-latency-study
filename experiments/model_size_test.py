from utils import benchmark_model, save_summary

def run_model_size_test():
    models = [
        {"path": "yolov8n.pt", "name": "yolov8n", "desc": "Nano model - optimized for edge"},
        {"path": "yolov8s.pt", "name": "yolov8s", "desc": "Small model - better accuracy"}
    ]

    resolution = "640x640"
    res_val = 640

    for m in models:
        print(f"Benchmarking model: {m['name']}")
        avg_lat, fps, is_half = benchmark_model(m['path'], resolution=res_val)
        precision = "FP16" if is_half else "FP32"

        save_summary(
            resolution=resolution,
            model_name=m['name'],
            precision=precision,
            fps=fps,
            latency=avg_lat,
            observation=m['desc']
        )

if __name__ == "__main__":
    run_model_size_test()
