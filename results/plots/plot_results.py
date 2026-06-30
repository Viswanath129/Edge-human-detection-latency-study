import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Use absolute path resolution relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "../tables/summary.csv")

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # Resolution Comparison (Nano model, FP32)
    res_df = df[(df['Model'] == 'yolov8n.pt') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        # Plot 1: Average Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_resolution.png'))
        plt.close()

        # Plot 2: Average FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.2, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'fps_vs_resolution.png'))
        plt.close()

    # Model Size Comparison (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color=['teal', 'coral'])
        plt.title('Latency vs Model Variant (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_model.png'))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color=['mediumpurple', 'gold'])
        plt.title('FPS vs Model Variant (640x640, FP32)')
        plt.xlabel('Model')
        plt.ylabel('Average FPS')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + 0.1, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'fps_vs_model.png'))
        plt.close()

    # Precision Comparison (YOLOv8n, 640x640)
    prec_df = df[(df['Model'] == 'yolov8n.pt') & (df['Resolution'] == '640x640')]
    if len(prec_df) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['gray', 'violet'])
        plt.title('Latency vs Precision (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig(os.path.join(script_dir, 'latency_vs_precision.png'))
        plt.close()

    print(f'Plots updated successfully in {script_dir}')

if __name__ == "__main__":
    generate_plots()
