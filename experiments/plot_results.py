import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Use absolute paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, "results", "tables", "summary.csv")
    plots_dir = os.path.join(base_dir, "results", "plots")

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Latency vs Resolution (for YOLOv8n, FP32)
    plt.figure(figsize=(10, 6))
    sub_df = df[(df["Model"] == "YOLOv8n") & (df["Precision"] == "FP32")]
    if not sub_df.empty:
        plt.bar(sub_df["Resolution"], sub_df["Average_Latency_ms"], color="salmon")
        plt.title("Inference Latency vs Input Resolution (YOLOv8n, FP32)")
        plt.xlabel("Resolution")
        plt.ylabel("Latency (ms)")
        for i, v in enumerate(sub_df["Average_Latency_ms"]):
            plt.text(i, v + 1, str(v), ha="center")
        plt.savefig(os.path.join(plots_dir, "latency_vs_resolution.png"))
    plt.close()

    # Plot 2: Latency vs Model (at 640x640, FP32)
    plt.figure(figsize=(10, 6))
    sub_df = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")]
    if not sub_df.empty:
        plt.bar(sub_df["Model"], sub_df["Average_Latency_ms"], color="lightblue")
        plt.title("Inference Latency vs Model Architecture (640x640, FP32)")
        plt.xlabel("Model")
        plt.ylabel("Latency (ms)")
        for i, v in enumerate(sub_df["Average_Latency_ms"]):
            plt.text(i, v + 1, str(v), ha="center")
        plt.savefig(os.path.join(plots_dir, "latency_vs_model.png"))
    plt.close()

    print(f"Updated plots saved to {plots_dir}")

if __name__ == "__main__":
    generate_plots()
