import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Generating benchmark plots...")

    # Locate summary.csv using absolute paths relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(current_dir, "..", "results", "plots"))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Error: summary.csv not found at {summary_path}")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary.csv: {e}")
        return

    print(f"Loaded summary with columns: {list(df.columns)}")
    print(df)

    # Filter standard models/resolutions for cleaner comparisons
    # Ensure columns match standard schema: Resolution, Model, Precision, Average_FPS, Average_Latency_ms
    # Convert types just in case
    df["Average_FPS"] = pd.to_numeric(df["Average_FPS"], errors="coerce")
    df["Average_Latency_ms"] = pd.to_numeric(df["Average_Latency_ms"], errors="coerce")

    # -------------------------------------------------------------
    # Plot 1 & 2: Resolution comparisons for YOLOv8n FP32
    # -------------------------------------------------------------
    res_df = df[(df["Model"] == "YOLOv8n") & (df["Precision"] == "FP32")]
    if not res_df.empty:
        # Sort by resolution (e.g., 416x416 then 640x640)
        res_df = res_df.sort_values(by="Resolution")

        # Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_Latency_ms"], color=["salmon", "lightblue"], width=0.4)
        plt.title("Average Inference Latency vs Input Resolution (YOLOv8n, FP32)")
        plt.xlabel("Input Resolution")
        plt.ylabel("Average Latency (ms)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(res_df["Average_Latency_ms"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_resolution.png"), dpi=150)
        plt.close()

        # FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_FPS"], color=["lightgreen", "orange"], width=0.4)
        plt.title("Average FPS vs Input Resolution (YOLOv8n, FP32)")
        plt.xlabel("Input Resolution")
        plt.ylabel("Average FPS (Frames/Sec)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(res_df["Average_FPS"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_resolution.png"), dpi=150)
        plt.close()

    # -------------------------------------------------------------
    # Plot 3 & 4: Model size comparisons (640x640, FP32)
    # -------------------------------------------------------------
    size_df = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")]
    if not size_df.empty:
        # Sort in order: yolov8n, yolov8s, yolov8m
        size_order = {"YOLOv8n": 0, "YOLOv8s": 1, "YOLOv8m": 2}
        size_df["sort_key"] = size_df["Model"].map(size_order)
        size_df = size_df.dropna(subset=["sort_key"]).sort_values(by="sort_key")

        # Latency vs Model Size
        plt.figure(figsize=(8, 5))
        plt.bar(size_df["Model"], size_df["Average_Latency_ms"], color=["cornflowerblue", "mediumaquamarine", "lightcoral"], width=0.5)
        plt.title("Inference Latency vs YOLOv8 Model Size (640x640, FP32)")
        plt.xlabel("Model Size")
        plt.ylabel("Average Latency (ms)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(size_df["Average_Latency_ms"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_model.png"), dpi=150)
        plt.close()

        # FPS vs Model Size
        plt.figure(figsize=(8, 5))
        plt.bar(size_df["Model"], size_df["Average_FPS"], color=["cornflowerblue", "mediumaquamarine", "lightcoral"], width=0.5)
        plt.title("Throughput (FPS) vs YOLOv8 Model Size (640x640, FP32)")
        plt.xlabel("Model Size")
        plt.ylabel("Average FPS (Frames/Sec)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(size_df["Average_FPS"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_model.png"), dpi=150)
        plt.close()

    # -------------------------------------------------------------
    # Plot 5 & 6: Precision comparison (640x640, YOLOv8n, FP32 vs FP16) - only if FP16 is present
    # -------------------------------------------------------------
    prec_df = df[(df["Resolution"] == "640x640") & (df["Model"] == "YOLOv8n")]
    if len(prec_df) >= 2 and "FP16" in prec_df["Precision"].values:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_Latency_ms"], color=["mediumpurple", "teal"], width=0.4)
        plt.title("Inference Latency Comparison: FP32 vs FP16 (YOLOv8n, 640x640)")
        plt.xlabel("Precision")
        plt.ylabel("Average Latency (ms)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(prec_df["Average_Latency_ms"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_precision.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_FPS"], color=["mediumpurple", "teal"], width=0.4)
        plt.title("Throughput (FPS) Comparison: FP32 vs FP16 (YOLOv8n, 640x640)")
        plt.xlabel("Precision")
        plt.ylabel("Average FPS (Frames/Sec)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, val in enumerate(prec_df["Average_FPS"]):
            if not pd.isna(val):
                plt.text(i, val + (val * 0.02), f"{val:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_precision.png"), dpi=150)
        plt.close()

    print("All comparative plots successfully generated and saved to results/plots/")

if __name__ == "__main__":
    main()
