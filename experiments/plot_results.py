import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    # Absolute path resolution relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(current_dir, "..", "results", "plots"))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Summary table not found at {summary_path}. Cannot generate plots.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary table: {e}")
        return

    if df.empty:
        print("Summary table is empty. Skipping plotting.")
        return

    print(f"Generating visualizations using summary data from {summary_path}...")

    # Plot 1: Average Latency vs Model Variant / Resolution
    # Create labels using Model + Resolution + Precision combinations
    df["label"] = df["Model"] + "\n" + df["Resolution"] + "\n" + df["Precision"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["label"], df["Average_Latency_ms"], color="salmon", width=0.5)
    plt.title("Inference Latency Comparison across YOLOv8 Configurations", fontsize=14, pad=15)
    plt.xlabel("Configuration (Model, Resolution, Precision)", fontsize=11, labelpad=10)
    plt.ylabel("Average Latency (ms)", fontsize=11, labelpad=10)
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (height * 0.02), f"{height:.1f}ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    latency_plot_path = os.path.join(plots_dir, "latency_vs_configuration.png")
    plt.savefig(latency_plot_path, dpi=150)
    plt.close()
    print(f"Saved latency plot to: {latency_plot_path}")

    # Plot 2: Average FPS Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["label"], df["Average_FPS"], color="lightgreen", width=0.5)
    plt.title("Inference FPS Comparison across YOLOv8 Configurations", fontsize=14, pad=15)
    plt.xlabel("Configuration (Model, Resolution, Precision)", fontsize=11, labelpad=10)
    plt.ylabel("Average Throughput (FPS)", fontsize=11, labelpad=10)
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (height * 0.02), f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fps_plot_path = os.path.join(plots_dir, "fps_vs_configuration.png")
    plt.savefig(fps_plot_path, dpi=150)
    plt.close()
    print(f"Saved FPS plot to: {fps_plot_path}")

if __name__ == "__main__":
    generate_plots()
