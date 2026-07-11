import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmarks():
    # Load the summary data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../results/tables/summary.csv")
    plots_dir = os.path.join(script_dir, "../results/plots")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Latency vs Resolution (Standardize on YOLOv8n, FP32)
    res_df = df[(df["Model"] == "yolov8n") & (df["Precision"] == "FP32")].sort_values("Resolution", ascending=False)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_Latency_ms"], color="skyblue")
        plt.title("Latency vs Input Resolution (YOLOv8n, FP32)")
        plt.ylabel("Latency (ms)")
        plt.savefig(os.path.join(plots_dir, "latency_vs_resolution.png"))
        plt.close()

    # 2. Latency vs Model Size (Standardize on 640x640, FP32)
    model_df = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")].sort_values("Average_Latency_ms")
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df["Model"], model_df["Average_Latency_ms"], color="salmon")
        plt.title("Latency vs Model Architecture (640x640, FP32)")
        plt.ylabel("Latency (ms)")
        plt.savefig(os.path.join(plots_dir, "latency_vs_model.png"))
        plt.close()

    # 3. Latency vs Precision (Standardize on 640x640, YOLOv8n)
    prec_df = df[(df["Resolution"] == "640x640") & (df["Model"] == "yolov8n")]
    if len(prec_df) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_Latency_ms"], color="lightgreen")
        plt.title("Latency vs Precision (YOLOv8n, 640x640)")
        plt.ylabel("Latency (ms)")
        plt.savefig(os.path.join(plots_dir, "latency_vs_precision.png"))
        plt.close()

    print(f"Plots updated in {plots_dir}")

if __name__ == "__main__":
    plot_benchmarks()
