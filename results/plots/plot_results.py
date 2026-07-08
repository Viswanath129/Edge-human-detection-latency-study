import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the summary data
    summary_path = '../tables/summary.csv'
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # 1. Latency vs Resolution (for yolov8n.pt, FP32)
    plt.figure(figsize=(10, 6))
    res_df = df[(df['Model'] == 'yolov8n.pt') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        res_df = res_df.sort_values('Resolution')
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
        plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_vs_resolution.png')
    plt.close()

    # 2. FPS vs Resolution (for yolov8n.pt, FP32)
    plt.figure(figsize=(10, 6))
    if not res_df.empty:
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig('fps_vs_resolution.png')
    plt.close()

    # 3. Latency vs Model Size (at 640x640, FP32)
    plt.figure(figsize=(10, 6))
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        model_df = model_df.sort_values('Average_Latency_ms')
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='salmon')
        plt.title('Inference Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_vs_model.png')
    plt.close()

    # 4. Latency vs Precision (for yolov8n.pt at 640x640)
    plt.figure(figsize=(10, 6))
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n.pt')]
    if len(prec_df) > 1:
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='orchid')
        plt.title('Inference Latency vs Precision (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_vs_precision.png')
    plt.close()

    print('Plots updated successfully in results/plots')

if __name__ == "__main__":
    main()
