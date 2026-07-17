import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Generating comparative visualization plots...")

    # Resolve absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "plots"))

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return

    # Normalize inputs for robust filtering
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Resolution"] = df["Resolution"].astype(str).str.strip()
    df["Precision"] = df["Precision"].astype(str).str.strip()

    print("Summary Data:")
    print(df)

    # 1. Plot Resolution Comparison (Model='YOLOv8n', Precision='FP32')
    res_df = df[(df["Model"] == "YOLOv8n") & (df["Precision"] == "FP32")]
    if not res_df.empty:
        # Sort so 416 is first or 640 is first
        res_df = res_df.sort_values(by="Resolution")

        # Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_Latency_ms"], color=["lightblue", "salmon"])
        plt.title("Inference Latency vs Input Resolution (YOLOv8n, FP32)")
        plt.xlabel("Input Resolution")
        plt.ylabel("Average Latency (ms)")
        for i, v in enumerate(res_df["Average_Latency_ms"]):
            plt.text(i, v + (max(res_df["Average_Latency_ms"])*0.02), f"{v:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_resolution.png"))
        plt.close()

        # FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_FPS"], color=["lightgreen", "orange"])
        plt.title("Throughput (FPS) vs Input Resolution (YOLOv8n, FP32)")
        plt.xlabel("Input Resolution")
        plt.ylabel("Average FPS (Frames/Sec)")
        for i, v in enumerate(res_df["Average_FPS"]):
            plt.text(i, v + (max(res_df["Average_FPS"])*0.02), f"{v:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_resolution.png"))
        plt.close()
        print("Generated resolution comparison plots.")
    else:
        print("Skipping resolution comparison plots (no matching data for YOLOv8n, FP32).")

    # 2. Plot Model Size Comparison (Resolution='640x640', Precision='FP32')
    model_df = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")]
    if not model_df.empty:
        model_df = model_df.sort_values(by="Model")

        # Latency vs Model Size
        plt.figure(figsize=(8, 5))
        plt.bar(model_df["Model"], model_df["Average_Latency_ms"], color=["salmon", "skyblue"])
        plt.title("Inference Latency vs Model Configuration (640x640, FP32)")
        plt.xlabel("Model Configuration")
        plt.ylabel("Average Latency (ms)")
        for i, v in enumerate(model_df["Average_Latency_ms"]):
            plt.text(i, v + (max(model_df["Average_Latency_ms"])*0.02), f"{v:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_model.png"))
        plt.close()

        # FPS vs Model Size
        plt.figure(figsize=(8, 5))
        plt.bar(model_df["Model"], model_df["Average_FPS"], color=["orange", "lightgreen"])
        plt.title("Throughput (FPS) vs Model Configuration (640x640, FP32)")
        plt.xlabel("Model Configuration")
        plt.ylabel("Average FPS (Frames/Sec)")
        for i, v in enumerate(model_df["Average_FPS"]):
            plt.text(i, v + (max(model_df["Average_FPS"])*0.02), f"{v:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_model.png"))
        plt.close()
        print("Generated model size comparison plots.")
    else:
        print("Skipping model size comparison plots (no matching data for 640x640, FP32).")

    # 3. Plot Precision Comparison (Model='YOLOv8n', Resolution='640x640')
    prec_df = df[(df["Model"] == "YOLOv8n") & (df["Resolution"] == "640x640")]
    # Plot only if both FP32 and FP16 exist
    if len(prec_df) >= 2 and "FP16" in prec_df["Precision"].values:
        prec_df = prec_df.sort_values(by="Precision", ascending=False)  # FP32, then FP16

        # Latency vs Precision
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_Latency_ms"], color=["lightblue", "violet"])
        plt.title("Inference Latency vs Precision Level (YOLOv8n, 640x640)")
        plt.xlabel("Precision Level")
        plt.ylabel("Average Latency (ms)")
        for i, v in enumerate(prec_df["Average_Latency_ms"]):
            plt.text(i, v + (max(prec_df["Average_Latency_ms"])*0.02), f"{v:.1f} ms", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "latency_vs_precision.png"))
        plt.close()

        # FPS vs Precision
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_FPS"], color=["lightgreen", "orchid"])
        plt.title("Throughput (FPS) vs Precision Level (YOLOv8n, 640x640)")
        plt.xlabel("Precision Level")
        plt.ylabel("Average FPS (Frames/Sec)")
        for i, v in enumerate(prec_df["Average_FPS"]):
            plt.text(i, v + (max(prec_df["Average_FPS"])*0.02), f"{v:.1f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "fps_vs_precision.png"))
        plt.close()
        print("Generated precision comparison plots.")
    else:
        print("Skipping precision comparison plots (FP16 data missing or not enough data points).")

    print(f"All generated visualization plots saved in: {plots_dir}")

if __name__ == "__main__":
    main()
