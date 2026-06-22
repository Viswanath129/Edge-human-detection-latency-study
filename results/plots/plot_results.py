import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_all():
    # Load the summary data
    if not os.path.exists('../tables/summary.csv'):
        print("Summary CSV not found. Run benchmarks first.")
        return

    df = pd.read_csv('../tables/summary.csv')

    # Ensure plots directory
    os.makedirs('.', exist_ok=True)

    # 1. Latency vs Resolution (for yolov8n, FP32)
    df_res = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not df_res.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Input Resolution (yolov8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Avg Latency (ms)')
        for i, v in enumerate(df_res['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_vs_resolution.png')
        plt.close()

    # 2. FPS vs Resolution (for yolov8n, FP32)
    if not df_res.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color='lightgreen')
        plt.title('FPS vs Input Resolution (yolov8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Avg FPS')
        for i, v in enumerate(df_res['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig('fps_vs_resolution.png')
        plt.close()

    # 3. Latency: yolov8n vs yolov8s (640x640, FP32)
    df_model = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if len(df_model) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(df_model['Model'], df_model['Average_Latency_ms'], color='skyblue')
        plt.title('Latency Comparison: Model Architecture (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Avg Latency (ms)')
        for i, v in enumerate(df_model['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_model_comp.png')
        plt.close()

    # 4. Latency: FP32 vs FP16 (yolov8n, 640x640)
    df_prec = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
    if len(df_prec) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(df_prec['Precision'], df_prec['Average_Latency_ms'], color='orchid')
        plt.title('Latency Comparison: FP32 vs FP16 (yolov8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Avg Latency (ms)')
        for i, v in enumerate(df_prec['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_precision_comp.png')
        plt.close()

    print('Plots updated and saved in results/plots/')

if __name__ == "__main__":
    plot_all()
