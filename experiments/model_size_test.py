from utils import run_benchmark, save_summary

def main():
    results = []

    # Nano
    res_n = run_benchmark("yolov8n.pt", imgsz=640, num_frames=10)
    results.append({
        "Resolution": "640x640",
        "Model": "YOLOv8n",
        "Precision": "FP32",
        "Average_FPS": round(res_n["fps"], 2),
        "Average_Latency_ms": round(res_n["avg_latency_ms"], 2),
        "Observation": "Fastest model"
    })

    # Small
    res_s = run_benchmark("yolov8s.pt", imgsz=640, num_frames=10)
    results.append({
        "Resolution": "640x640",
        "Model": "YOLOv8s",
        "Precision": "FP32",
        "Average_FPS": round(res_s["fps"], 2),
        "Average_Latency_ms": round(res_s["avg_latency_ms"], 2),
        "Observation": "Balanced accuracy/speed"
    })

    save_summary(results)

if __name__ == "__main__":
    main()
