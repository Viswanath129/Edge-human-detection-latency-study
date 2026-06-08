import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots():
    # Path to summary file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../tables/summary.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmarks first.")
        return

    # Load the summary data
    df = pd.read_csv(csv_path)

    # Set output directory to current file's directory
    os.chdir(base_dir)

    # Plot 1: Average Latency vs Resolution (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig('latency_vs_resolution.png')
        plt.close()

    # Plot 2: Average FPS vs Resolution (YOLOv8n, FP32)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.2, f"{v:.1f}", ha='center')
        plt.savefig('fps_vs_resolution.png')
        plt.close()

    # Plot 3: Latency: FP32 vs FP16 (YOLOv8n, 640x640)
    prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
    if len(prec_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['teal', 'orchid'])
        plt.title('Latency Comparison: FP32 vs FP16 (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.savefig('latency_precision_comp.png')
        plt.close()

    # Plot 4: FPS vs Model Size (640x640, FP32)
    size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if len(size_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(size_df['Model'], size_df['Average_FPS'], color=['gold', 'skyblue'])
        plt.title('FPS vs Model Size (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average FPS')
        for i, v in enumerate(size_df['Average_FPS']):
            plt.text(i, v + 0.2, f"{v:.1f}", ha='center')
        plt.savefig('fps_vs_model.png')
        plt.close()

    print(f"Plots saved successfully in {base_dir}")

if __name__ == "__main__":
    generate_plots()
