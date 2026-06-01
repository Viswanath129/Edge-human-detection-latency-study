import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Absolute path to summary.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "../tables/summary.csv")

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)

    # Plot 1: Resolution Comparison (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='skyblue')
        plt.title('FPS vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('Average FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'fps_vs_resolution.png'))
        plt.close()

    # Plot 2: Model Size Comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color='lightgreen')
        plt.title('FPS vs Model Variant (640x640, FP32)')
        plt.ylabel('Average FPS')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'fps_vs_model.png'))
        plt.close()

    # Plot 3: Precision Comparison (YOLOv8n, 640x640)
    prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
    if not prec_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Precision (YOLOv8n, 640x640)')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 1, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'latency_vs_precision.png'))
        plt.close()

    print(f"Plots generated in {base_dir}")

if __name__ == "__main__":
    generate_plots()
