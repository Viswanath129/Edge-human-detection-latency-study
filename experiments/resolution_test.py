from utils import run_benchmark, save_summary

def main():
    results = []

    # 640
    res_640 = run_benchmark("yolov8n.pt", imgsz=640, num_frames=10)
    results.append({
        "Resolution": "640x640",
        "Model": "YOLOv8n",
        "Precision": "FP32",
        "Average_FPS": round(res_640["fps"], 2),
        "Average_Latency_ms": round(res_640["avg_latency_ms"], 2),
        "Observation": "Standard resolution"
    })

    # 416
    res_416 = run_benchmark("yolov8n.pt", imgsz=416, num_frames=10)
    results.append({
        "Resolution": "416x416",
        "Model": "YOLOv8n",
        "Precision": "FP32",
        "Average_FPS": round(res_416["fps"], 2),
        "Average_Latency_ms": round(res_416["avg_latency_ms"], 2),
        "Observation": "Reduced latency"
    })

    save_summary(results)

if __name__ == "__main__":
    main()
