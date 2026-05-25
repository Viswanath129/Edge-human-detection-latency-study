from utils import run_benchmark, save_summary

def main():
    results = []

    # FP32 (Shared with resolution test, but good for completeness)
    res_fp32 = run_benchmark("yolov8n.pt", imgsz=640, half=False, num_frames=10)
    results.append({
        "Resolution": "640x640",
        "Model": "YOLOv8n",
        "Precision": "FP32",
        "Average_FPS": round(res_fp32["fps"], 2),
        "Average_Latency_ms": round(res_fp32["avg_latency_ms"], 2),
        "Observation": "Baseline precision"
    })

    # FP16
    res_fp16 = run_benchmark("yolov8n.pt", imgsz=640, half=True, num_frames=10)
    results.append({
        "Resolution": "640x640",
        "Model": "YOLOv8n",
        "Precision": "FP16",
        "Average_FPS": round(res_fp16["fps"], 2),
        "Average_Latency_ms": round(res_fp16["avg_latency_ms"], 2),
        "Observation": "Hardware acceleration potential"
    })

    save_summary(results)

if __name__ == "__main__":
    main()
