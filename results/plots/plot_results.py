import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    # Find summary.csv relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "tables", "summary.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Ensure plots directory exists (should be script_dir)
    os.makedirs(script_dir, exist_ok=True)

    # 1. Latency vs Resolution (for yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Resolution')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_resolution.png'))
        plt.close()

    # 2. FPS vs Resolution (for yolov8n, FP32)
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('Throughput (FPS) vs Input Resolution (YOLOv8n, FP32)')
        plt.ylabel('FPS')
        plt.xlabel('Resolution')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 3. Latency vs Model Size (at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Inference Latency vs Model Size (640x640, FP32)')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Model')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_model.png'))
        plt.close()

    # 4. Latency vs Precision (at 640x640, yolov8n)
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
    if not prec_df.empty and len(prec_df) > 1:
        plt.figure(figsize=(10, 6))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='orchid')
        plt.title('Inference Latency vs Precision (640x640, YOLOv8n)')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Precision')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_precision.png'))
        plt.close()

    print(f"Plots updated in {script_dir}")

if __name__ == "__main__":
    plot_results()
