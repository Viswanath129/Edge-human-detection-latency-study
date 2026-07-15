import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Paths are relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "results", "tables", "summary.csv")
    output_dir = os.path.join(script_dir, "..", "results", "plots")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Latency vs Resolution (Filtering for YOLOv8n, FP32)
    plt.figure(figsize=(10, 6))
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
    if not res_df.empty:
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
        plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 1, str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_vs_resolution.png'))
    plt.close()

    # Plot 2: Latency vs Model (Filtering for 640x640, FP32)
    plt.figure(figsize=(10, 6))
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='lightblue')
        plt.title('Inference Latency vs Model Architecture (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 1, str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_vs_model.png'))
    plt.close()

    print(f"Plots updated in {output_dir}")

if __name__ == "__main__":
    generate_plots()
