import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Generating benchmark plots...")

    # Absolute paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../results/tables/summary.csv"))
    plots_dir = os.path.abspath(os.path.join(script_dir, "../results/plots"))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Summary file not found at {csv_path}. Cannot plot results.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded summary data:\n{df}")

    # Standardize columns
    cols = ["Resolution", "Model", "Precision", "Average_FPS", "Average_Latency_ms", "Observation"]
    df = df.reindex(columns=cols)

    # 1. Latency & FPS vs Resolution (YOLOv8n, FP32)
    df_res = df[(df["Model"] == "YOLOv8n") & (df["Precision"] == "FP32")]
    if not df_res.empty:
        # Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_res['Average_Latency_ms']):
            plt.text(i, v + 2, f"{v:.2f}", ha='center')
        latency_plot_path = os.path.join(plots_dir, 'latency_vs_resolution.png')
        plt.savefig(latency_plot_path)
        plt.close()
        print(f"Saved: {latency_plot_path}")

        # FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(df_res['Average_FPS']):
            plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
        fps_plot_path = os.path.join(plots_dir, 'fps_vs_resolution.png')
        plt.savefig(fps_plot_path)
        plt.close()
        print(f"Saved: {fps_plot_path}")

    # 2. Latency vs Model size (640x640, FP32)
    df_model = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")]
    if not df_model.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_model['Model'], df_model['Average_Latency_ms'], color=['skyblue', 'lightcoral'])
        plt.title('Average Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_model['Average_Latency_ms']):
            plt.text(i, v + 2, f"{v:.2f}", ha='center')
        model_plot_path = os.path.join(plots_dir, 'latency_vs_model.png')
        plt.savefig(model_plot_path)
        plt.close()
        print(f"Saved: {model_plot_path}")

    # 3. Latency vs Precision (if both FP32 and FP16 exist for YOLOv8n, 640x640)
    df_prec = df[(df["Resolution"] == "640x640") & (df["Model"] == "YOLOv8n")]
    if len(df_prec) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(df_prec['Precision'], df_prec['Average_Latency_ms'], color=['lightgray', 'gold'])
        plt.title('Average Latency vs Precision Level (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_prec['Average_Latency_ms']):
            plt.text(i, v + 2, f"{v:.2f}", ha='center')
        prec_plot_path = os.path.join(plots_dir, 'latency_vs_precision.png')
        plt.savefig(prec_plot_path)
        plt.close()
        print(f"Saved: {prec_plot_path}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
