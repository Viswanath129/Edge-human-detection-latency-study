import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the summary data
    summary_path = os.path.join(os.path.dirname(__file__), '../tables/summary.csv')
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)

    # Ensure plots directory exists
    plots_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Resolution Comparison (yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32') &
                (df['Resolution'].isin(['640x640', '416x416']))]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Latency vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('FPS vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 2. Precision Comparison (yolov8n, 640x640)
    prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
    if not prec_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Precision (YOLOv8n, 640x640)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_precision.png'))
        plt.close()

    # 3. Model Size Comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='orchid')
        plt.title('Latency vs Model Size (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model_size.png'))
        plt.close()

    print(f"Plots saved successfully in {plots_dir}")

if __name__ == "__main__":
    main()
