from utils import benchmark_model, save_summary

def run_resolution_test():
    resolutions = [
        {"val": 640, "str": "640x640", "desc": "Higher detection quality"},
        {"val": 416, "str": "416x416", "desc": "Faster Inference"}
    ]

    model_path = "yolov8n.pt"
    model_name = "yolov8n"

    for res in resolutions:
        print(f"Benchmarking resolution: {res['str']}")
        avg_lat, fps, is_half = benchmark_model(model_path, resolution=res['val'])
        precision = "FP16" if is_half else "FP32"

        save_summary(
            resolution=res['str'],
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_lat,
            observation=res['desc']
        )

if __name__ == "__main__":
    run_resolution_test()
