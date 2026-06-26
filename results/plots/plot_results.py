import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tables/summary.csv"))
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    plots_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Latency vs Resolution (yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

    # 2. FPS vs Resolution (yolov8n, FP32)
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('Throughput (FPS) vs Resolution (YOLOv8n, FP32)')
        plt.ylabel('FPS')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 3. Model Size Comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency Comparison by Model Size (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
        plt.close()

    # 4. Precision Comparison (if multiple precisions exist for 640x640 yolov8n)
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
    if len(prec_df) > 1:
        plt.figure(figsize=(10, 6))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='plum')
        plt.title('Latency Comparison: FP32 vs FP16 (640x640, YOLOv8n)')
        plt.ylabel('Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'latency_precision_comp.png'))
        plt.close()

    print(f"Plots generated in {plots_dir}")

if __name__ == "__main__":
    generate_plots()
