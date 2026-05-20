import pandas as pd
from utils import run_benchmark
import os

def main():
    resolutions = [640, 416]
    results = []

    for res in resolutions:
        res_results = run_benchmark(
            model_path="yolov8n.pt",
            img_size=res,
            experiment_name=f"resolution_{res}"
        )
        results.append(res_results)

    # Save aggregate results
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/resolution_summary.csv", index=False)
    print("\nResolution benchmark complete. Summary saved to results/tables/resolution_summary.csv")

if __name__ == "__main__":
    main()
