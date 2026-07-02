import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    summary_path = os.path.join(base_dir, "results", "tables", "summary.csv")
    plot_dir = os.path.join(base_dir, "results", "plots")

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Latency vs Resolution (for yolov8n.pt)
    res_df = df[df["Model"] == "yolov8n.pt"].copy()
    res_df["Resolution_Str"] = res_df["Resolution"].astype(str)
    res_df = res_df.sort_values("Resolution", ascending=False)

    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df["Resolution_Str"], res_df["Average_Latency_ms"], color='skyblue')
        plt.title("Inference Latency vs Resolution (YOLOv8n)")
        plt.xlabel("Resolution")
        plt.ylabel("Latency (ms)")
        plt.savefig(os.path.join(plot_dir, "latency_vs_resolution.png"))
        plt.close()

    # 2. FPS vs Model (for 640x640)
    model_df = df[df["Resolution"] == "640x640"].copy()
    model_df = model_df.dropna(subset=["Model"])
    model_df["Model_Str"] = model_df["Model"].astype(str)
    model_df = model_df.sort_values("Average_FPS")

    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df["Model_Str"], model_df["Average_FPS"], color='salmon')
        plt.title("FPS vs Model Architecture (640x640)")
        plt.xlabel("Model")
        plt.ylabel("FPS")
        plt.savefig(os.path.join(plot_dir, "fps_vs_model.png"))
        plt.close()

    print(f"Plots generated in {plot_dir}")

if __name__ == "__main__":
    plot_results()
