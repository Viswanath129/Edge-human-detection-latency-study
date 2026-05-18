import os
from utils import run_benchmark, save_results

def main():
    model_variants = ["yolov8n.pt", "yolov8s.pt"]
    headless = os.environ.get("HEADLESS", "false").lower() == "true"
    summary_data = []
    all_raw_latencies = {}

    for variant in model_variants:
        print(f"Benchmarking model variant: {variant}")
        avg_latency, fps, latencies = run_benchmark(variant, headless=headless)

        summary_data.append({
            "model_variant": variant,
            "avg_latency_ms": avg_latency,
            "fps": fps
        })
        all_raw_latencies[variant] = latencies
        print(f"{variant} - Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

    save_results("model_size_results.csv", summary_data, all_raw_latencies)

if __name__ == "__main__":
    main()
