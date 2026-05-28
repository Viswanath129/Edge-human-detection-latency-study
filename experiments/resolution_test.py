from utils import benchmark_model, save_summary

def run_resolution_test():
    results = []

    # Test 640x640
    avg_latency_640, fps_640 = benchmark_model("yolov8n.pt", input_size=640, save_path="results/tables/resolution_640_results.csv")
    results.append({
        "Resolution": "640x640",
        "Model": "yolov8n",
        "Precision": "FP32",
        "Average_FPS": round(fps_640, 2),
        "Average_Latency_ms": round(avg_latency_640, 2),
        "Observation": "Baseline high resolution"
    })

    # Test 416x416
    avg_latency_416, fps_416 = benchmark_model("yolov8n.pt", input_size=416, save_path="results/tables/resolution_416_results.csv")
    results.append({
        "Resolution": "416x416",
        "Model": "yolov8n",
        "Precision": "FP32",
        "Average_FPS": round(fps_416, 2),
        "Average_Latency_ms": round(avg_latency_416, 2),
        "Observation": "Faster inference, reduced detail"
    })

    return results

if __name__ == "__main__":
    test_results = run_resolution_test()
    # Note: summary is saved in a final step to include all data if needed,
    # but for now we can save it here or in a master script.
    # save_summary(test_results)
