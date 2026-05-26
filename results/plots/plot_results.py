import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Set up paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "../tables/summary.csv")
    output_dir = script_dir

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # 1. Resolution Comparison (yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
        plt.title('Latency vs Resolution')
        plt.ylabel('ms')

        plt.subplot(1, 2, 2)
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('FPS vs Resolution')
        plt.ylabel('FPS')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resolution_comparison.png'))
        plt.close()

    # 2. Model Size Comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='skyblue')
        plt.title('Latency vs Model Size')
        plt.ylabel('ms')

        plt.subplot(1, 2, 2)
        plt.bar(model_df['Model'], model_df['Average_FPS'], color='gold')
        plt.title('FPS vs Model Size')
        plt.ylabel('FPS')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_size_comparison.png'))
        plt.close()

    # 3. Precision Comparison (yolov8n, 640x640)
    prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
    if not prec_df.empty:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='plum')
        plt.title('Latency vs Precision')
        plt.ylabel('ms')

        plt.subplot(1, 2, 2)
        plt.bar(prec_df['Precision'], prec_df['Average_FPS'], color='lightcoral')
        plt.title('FPS vs Precision')
        plt.ylabel('FPS')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_comparison.png'))
        plt.close()

    print(f"Plots successfully generated in {output_dir}")

if __name__ == "__main__":
    generate_plots()
