import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Setup paths relative to the script location
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(exp_dir)
    csv_path = os.path.join(project_root, 'results', 'tables', 'summary.csv')
    plot_dir = os.path.join(project_root, 'results', 'plots')

    if not os.path.exists(csv_path):
        print(f"Summary CSV not found at {csv_path}")
        return

    # Load the summary data
    df = pd.read_csv(csv_path)

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Latency vs Resolution (for yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'latency_vs_resolution.png'))
        plt.close()

    # 2. FPS vs Resolution (for yolov8n, FP32)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('Throughput vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 3. Latency vs Model (at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')].sort_values('Average_Latency_ms')
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'latency_vs_model.png'))
        plt.close()

    # 4. Latency vs Precision (at 640x640, yolov8n)
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')].sort_values('Precision')
    if len(prec_df) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='orchid')
        plt.title('Latency vs Precision (640x640, YOLOv8n)')
        plt.xlabel('Precision')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig(os.path.join(plot_dir, 'latency_vs_precision.png'))
        plt.close()

    print(f"Plots updated successfully in {plot_dir}")

if __name__ == "__main__":
    main()
