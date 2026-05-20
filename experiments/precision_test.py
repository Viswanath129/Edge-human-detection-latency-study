import pandas as pd
from utils import run_benchmark
import os

def main():
    # Comparing FP32 and FP16 precision
    precisions = [False, True]  # False = FP32, True = FP16
    results = []

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        res_results = run_benchmark(
            model_path="yolov8n.pt",
            img_size=640,
            half=half,
            experiment_name=f"precision_{precision_name}"
        )
        # Add precision info to results
        res_results["precision"] = precision_name
        results.append(res_results)

    # Save aggregate results
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/precision_summary.csv", index=False)
    print("\nPrecision benchmark complete. Summary saved to results/tables/precision_summary.csv")

if __name__ == "__main__":
    main()
