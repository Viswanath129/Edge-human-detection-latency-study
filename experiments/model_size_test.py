import pandas as pd
from utils import run_benchmark
import os

def main():
    # Comparing YOLOv8 nano and small models
    models = ["yolov8n.pt", "yolov8s.pt"]
    results = []

    for model in models:
        model_name = model.split('.')[0]
        res_results = run_benchmark(
            model_path=model,
            img_size=640,
            experiment_name=f"model_{model_name}"
        )
        results.append(res_results)

    # Save aggregate results
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/model_size_summary.csv", index=False)
    print("\nModel size benchmark complete. Summary saved to results/tables/model_size_summary.csv")

if __name__ == "__main__":
    main()
