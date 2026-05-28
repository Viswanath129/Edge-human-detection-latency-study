from utils import benchmark_model

def run_precision_test():
    results = []

    # Test FP32
    avg_latency_32, fps_32 = benchmark_model("yolov8n.pt", input_size=640, half=False, save_path="results/tables/precision_fp32_results.csv")
    results.append({
        "Resolution": "640x640",
        "Model": "yolov8n",
        "Precision": "FP32",
        "Average_FPS": round(fps_32, 2),
        "Average_Latency_ms": round(avg_latency_32, 2),
        "Observation": "Standard precision"
    })

    # Test FP16
    avg_latency_16, fps_16 = benchmark_model("yolov8n.pt", input_size=640, half=True, save_path="results/tables/precision_fp16_results.csv")
    results.append({
        "Resolution": "640x640",
        "Model": "yolov8n",
        "Precision": "FP16",
        "Average_FPS": round(fps_16, 2),
        "Average_Latency_ms": round(avg_latency_16, 2),
        "Observation": "Half precision (optimized for GPU)"
    })

    return results

if __name__ == "__main__":
    run_precision_test()
