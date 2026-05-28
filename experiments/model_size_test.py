from utils import benchmark_model

def run_model_size_test():
    results = []

    # Test Nano (yolov8n)
    avg_latency_n, fps_n = benchmark_model("yolov8n.pt", input_size=640, save_path="results/tables/model_nano_results.csv")
    results.append({
        "Resolution": "640x640",
        "Model": "yolov8n",
        "Precision": "FP32",
        "Average_FPS": round(fps_n, 2),
        "Average_Latency_ms": round(avg_latency_n, 2),
        "Observation": "Nano model - fastest"
    })

    # Test Small (yolov8s)
    avg_latency_s, fps_s = benchmark_model("yolov8s.pt", input_size=640, save_path="results/tables/model_small_results.csv")
    results.append({
        "Resolution": "640x640",
        "Model": "yolov8s",
        "Precision": "FP32",
        "Average_FPS": round(fps_s, 2),
        "Average_Latency_ms": round(avg_latency_s, 2),
        "Observation": "Small model - better accuracy"
    })

    return results

if __name__ == "__main__":
    run_model_size_test()
