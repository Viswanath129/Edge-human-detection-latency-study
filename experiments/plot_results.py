import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    # Use absolute path to find summary.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "results", "tables", "summary.csv")
    plot_dir = os.path.join(base_dir, "results", "plots")

    if not os.path.exists(csv_path):
        print(f"Summary file not found at {csv_path}. Skipping plotting.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Latency vs Resolution (for YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs. Input Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Resolution')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'latency_vs_resolution.png'))
        plt.close()

    # 2. Latency vs Model Size (at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='lightcoral')
        plt.title('Inference Latency vs. Model Architecture (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Model')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'latency_vs_model.png'))
        plt.close()

    # 3. FPS Comparison (Overview)
    plt.figure(figsize=(12, 7))
    labels = [f"{row['Model']}\n{row['Resolution']}\n({row['Precision']})" for _, row in df.iterrows()]
    plt.bar(labels, df['Average_FPS'], color='seagreen')
    plt.title('End-to-End Throughput (FPS) Comparison')
    plt.ylabel('FPS')
    plt.xticks(rotation=45)
    for i, v in enumerate(df['Average_FPS']):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fps_comparison.png'))
    plt.close()

    print(f"Plots updated in {plot_dir}")

if __name__ == "__main__":
    plot_results()
