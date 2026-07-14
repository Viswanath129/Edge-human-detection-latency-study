import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Setup paths relative to script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(base_dir, 'results', 'tables', 'summary.csv')
    output_dir = os.path.join(base_dir, 'results', 'plots')

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # Plot 1: Latency vs Resolution (for YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_vs_resolution.png'))
        plt.close()

    # Plot 2: Latency vs Model (at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Inference Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_vs_model.png'))
        plt.close()

    print(f"Plots updated in {output_dir}")

if __name__ == "__main__":
    generate_plots()
