import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Load the summary data
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, '../tables/summary.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 1. Average Latency vs Resolution (for YOLOv8n, FP32)
    plt.figure(figsize=(8, 5))
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig(os.path.join(base_path, 'latency_vs_resolution.png'))
    plt.close()

    # 2. Average FPS vs Resolution (for YOLOv8n, FP32)
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS (Frames/Sec)')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.savefig(os.path.join(base_path, 'fps_vs_resolution.png'))
    plt.close()

    # 3. Latency: FP32 vs FP16 (at 640x640, YOLOv8n)
    plt.figure(figsize=(8, 5))
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')]
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Latency Comparison: FP32 vs FP16 (640x640, YOLOv8n)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig(os.path.join(base_path, 'latency_precision_comp.png'))
    plt.close()

    # 4. FPS: YOLOv8n vs YOLOv8s (at 640x640, FP32)
    plt.figure(figsize=(8, 5))
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['gold', 'skyblue'])
    plt.title('FPS Comparison: Nano vs Small (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.savefig(os.path.join(base_path, 'fps_vs_model.png'))
    plt.close()

    print('All plots generated successfully in results/plots/')

if __name__ == "__main__":
    generate_plots()
