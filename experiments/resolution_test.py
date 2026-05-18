import os
from utils import run_benchmark, save_results

def main():
    resolutions = [640, 416]
    headless = os.environ.get("HEADLESS", "false").lower() == "true"
    summary_data = []
    all_raw_latencies = {}

    for res in resolutions:
        print(f"Benchmarking resolution: {res}x{res}")
        avg_latency, fps, latencies = run_benchmark("yolov8n.pt", imgsz=res, headless=headless)

        summary_data.append({
            "resolution": res,
            "avg_latency_ms": avg_latency,
            "fps": fps
        })
        all_raw_latencies[f"res_{res}"] = latencies
        print(f"{res}x{res} - Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

    save_results("resolution_results.csv", summary_data, all_raw_latencies)

if __name__ == "__main__":
    main()
