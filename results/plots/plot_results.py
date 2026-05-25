import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the summary data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, '../tables/summary.csv')
    df = pd.read_csv(summary_path)

    # Filter for Resolution comparison (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]

    # Plot 1: Average Latency vs Resolution
    plt.figure(figsize=(10, 6))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(script_dir, 'latency_vs_resolution.png'))
    plt.close()

    # Plot 2: Average FPS vs Resolution
    plt.figure(figsize=(10, 6))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Frames Per Second vs Input Resolution (YOLOv8n, FP32)')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(script_dir, 'fps_vs_resolution.png'))
    plt.close()

    # Filter for Model Size comparison (640x640, FP32)
    size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]

    # Plot 3: Latency vs Model Size
    plt.figure(figsize=(10, 6))
    plt.bar(size_df['Model'], size_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Inference Latency vs Model Size (640x640, FP32)')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(size_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(script_dir, 'latency_vs_model_size.png'))
    plt.close()

    # Filter for Precision comparison (640x640, YOLOv8n)
    # Note: FP16 might be extremely slow on CPU, we might want to plot it on a log scale or just acknowledge it
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')]

    # Plot 4: Latency vs Precision
    plt.figure(figsize=(10, 6))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['gold', 'silver'])
    plt.title('Inference Latency vs Precision (640x640, YOLOv8n)')
    plt.ylabel('Average Latency (ms)')
    plt.yscale('log') # Use log scale because FP16 is so slow on CPU
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(script_dir, 'latency_vs_precision.png'))
    plt.close()

    print('All plots generated successfully in results/plots/')

if __name__ == "__main__":
    main()
