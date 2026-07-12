import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, '../results/tables/summary.csv')
    plots_dir = os.path.join(base_dir, '../results/plots')

    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    os.makedirs(plots_dir, exist_ok=True)

    # Filter for Resolution comparison (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

    # Filter for Model Size comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='lightgreen')
        plt.title('Inference Latency vs Model Size (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
        plt.close()

    print(f"Plots updated in {plots_dir}")

if __name__ == "__main__":
    generate_plots()
