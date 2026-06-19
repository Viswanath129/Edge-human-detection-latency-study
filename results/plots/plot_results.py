import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Use absolute path relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../tables/summary.csv')

    if not os.path.exists(csv_path):
        print(f"Summary CSV not found at {csv_path}. Skipping plot generation.")
        return

    # Load the summary data
    df = pd.read_csv(csv_path)

    # Output directory for plots
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    # Filter for Resolution Test (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        # Plot 1: Average Latency vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + (v*0.02), str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_vs_resolution.png'))
        plt.close()

        # Plot 2: Average FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + (v*0.02), str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'fps_vs_resolution.png'))
        plt.close()

    # Filter for Model Size Test (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if len(model_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color=['skyblue', 'steelblue'])
        plt.title('Inference Speed Comparison: YOLOv8n vs YOLOv8s (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average FPS')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + (v*0.02), str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'fps_vs_model.png'))
        plt.close()

    # Filter for Precision Test (YOLOv8n, 640x640)
    prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
    if len(prec_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['grey', 'gold'])
        plt.title('Precision Impact: FP32 vs FP16 Latency (YOLOv8n, 640x640)')
        plt.xlabel('Precision Level')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + (v*0.02), str(v), ha='center')
        plt.savefig(os.path.join(output_dir, 'latency_precision_comp.png'))
        plt.close()

    print(f'Plots updated successfully in {output_dir}')

if __name__ == "__main__":
    generate_plots()
