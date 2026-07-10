import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "plots"))

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}. Skipping plotting.")
        return

    df = pd.read_csv(summary_path)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Latency vs Resolution (for YOLOv8n, FP32)
    plt.figure(figsize=(10, 6))
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        res_df = res_df.sort_values('Resolution', ascending=False)
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Latency vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 1, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
    plt.close()

    # 2. Latency vs Model (at 640x640, FP32)
    plt.figure(figsize=(10, 6))
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Model Size (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 1, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
    plt.close()

    print(f"Plots updated in {plots_dir}")

if __name__ == "__main__":
    plot_results()
