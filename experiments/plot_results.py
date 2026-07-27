import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    # Resolve absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, '../results/tables/summary.csv'))
    plots_dir = os.path.abspath(os.path.join(current_dir, '../results/plots'))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Summary CSV not found at {summary_path}")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary CSV: {e}")
        return

    if df.empty:
        print("Summary CSV is empty")
        return

    # Standardized columns: Resolution, Model, Precision, Average_FPS, Average_Latency_ms, Observation

    # 1. Plot Latency vs Resolution (for YOLOv8n, FP32)
    df_res = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not df_res.empty:
        df_res = df_res.drop_duplicates(subset=['Resolution'])
        # Sort so 416 comes before 640
        df_res = df_res.sort_values(by='Resolution')

        # Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['lightblue', 'salmon'], width=0.4)
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_res['Average_Latency_ms']):
            plt.text(i, v + (max(df_res['Average_Latency_ms']) * 0.02), f"{v:.2f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

        # FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['orange', 'lightgreen'], width=0.4)
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(df_res['Average_FPS']):
            plt.text(i, v + (max(df_res['Average_FPS']) * 0.02), f"{v:.2f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
        plt.close()

    # 2. Plot Latency vs Model Size (at 640x640, FP32)
    df_model = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not df_model.empty:
        df_model = df_model.drop_duplicates(subset=['Model'])
        df_model = df_model.sort_values(by='Model')

        plt.figure(figsize=(8, 5))
        plt.bar(df_model['Model'], df_model['Average_Latency_ms'], color=['lightskyblue', 'lightcoral'], width=0.4)
        plt.title('Average Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Size')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_model['Average_Latency_ms']):
            plt.text(i, v + (max(df_model['Average_Latency_ms']) * 0.02), f"{v:.2f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
        plt.close()

    print("Plots updated successfully in results/plots/")

if __name__ == '__main__':
    generate_plots()
