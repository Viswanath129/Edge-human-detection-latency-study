import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    # Correct path to summary.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, '../tables/summary.csv')

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # Ensure plots directory exists
    os.makedirs(base_dir, exist_ok=True)

    # 1. Resolution Comparison (Nano model, FP32)
    res_df = df[(df['Model'] == 'Nano') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='skyblue')
        plt.title('Inference Speed (FPS) vs Resolution (Nano, FP32)')
        plt.ylabel('FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 2. Precision Comparison (Nano model, 640x640)
    prec_df = df[(df['Model'] == 'Nano') & (df['Resolution'] == '640x640')]
    if not prec_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='lightcoral')
        plt.title('Latency (ms) vs Precision (Nano, 640x640)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'latency_vs_precision.png'))
        plt.close()

    # 3. Model Size Comparison (640x640, FP32)
    size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not size_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(size_df['Model'], size_df['Average_FPS'], color='lightgreen')
        plt.title('Inference Speed (FPS) vs Model Variant (640x640, FP32)')
        plt.ylabel('FPS')
        for i, v in enumerate(size_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(base_dir, 'fps_vs_model_size.png'))
        plt.close()

    print(f"Plots updated in {base_dir}")

if __name__ == "__main__":
    plot_results()
