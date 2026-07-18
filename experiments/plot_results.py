import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    # Use absolute paths relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(current_dir, "..", "results", "plots"))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Summary CSV file not found at {summary_path}. Skipping plot generation.")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        print("Summary CSV is empty. Skipping plot generation.")
        return

    print("Loaded summary data for plotting:")
    print(df)

    # Ensure columns exist and handle case-insensitivity just in case
    # Resolution,Model,Precision,Average_FPS,Average_Latency_ms,Observation

    # Plot 1: Average Latency vs Resolution (for FP32, YOLOv8n)
    df_res = df[(df['Model'].str.lower() == 'yolov8n') & (df['Precision'].str.upper() == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
    if not df_res.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['salmon', 'lightblue'], width=0.4)
        plt.title('Average Latency vs Input Resolution (YOLOv8n FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(df_res['Average_Latency_ms']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'latency_vs_resolution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

    # Plot 2: Average FPS vs Resolution (for FP32, YOLOv8n)
    if not df_res.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['lightgreen', 'orange'], width=0.4)
        plt.title('Average FPS vs Input Resolution (YOLOv8n FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(df_res['Average_FPS']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'fps_vs_resolution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

    # Plot 3: Average Latency vs Model (YOLOv8n vs YOLOv8s at 640x640 resolution)
    df_model = df[(df['Resolution'] == '640x640') & (df['Precision'].str.upper() == 'FP32')]
    if not df_model.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_model['Model'], df_model['Average_Latency_ms'], color=['teal', 'coral'], width=0.4)
        plt.title('Average Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Architecture')
        plt.ylabel('Average Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(df_model['Average_Latency_ms']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'latency_vs_model.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_path}")

if __name__ == "__main__":
    generate_plots()
