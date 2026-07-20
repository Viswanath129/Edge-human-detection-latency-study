import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Setup absolute paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(script_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "plots"))

    if not os.path.exists(summary_path):
        print(f"Summary file not found at: {summary_path}")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Load summary data
    df = pd.read_csv(summary_path)

    # Ensure required columns exist
    # If the CSV is empty or missing columns, print and return
    required_cols = {'Resolution', 'Model', 'Precision', 'Average_FPS', 'Average_Latency_ms'}
    if not required_cols.issubset(df.columns):
        print(f"Missing required columns. Found: {list(df.columns)}")
        return

    print("Generating performance visualization plots...")

    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Latency & FPS vs Resolution (YOLOv8n, FP32)
    # Filter for Resolution Comparison
    res_df = df[(df['Model'].str.lower() == 'yolov8n') & (df['Precision'].str.upper() == 'FP32')]
    if not res_df.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:red'
        ax1.set_xlabel('Input Resolution')
        ax1.set_ylabel('Average Latency (ms)', color=color)
        bars1 = ax1.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=color, alpha=0.6, width=0.4, label='Latency (ms)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.bar_label(bars1, fmt='%.1f ms', padding=3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Average FPS', color=color)
        bars2 = ax2.bar(res_df['Resolution'], res_df['Average_FPS'], color=color, alpha=0.6, width=0.2, label='FPS')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.bar_label(bars2, fmt='%.1f FPS', padding=3)

        plt.title('Performance Comparison by Input Resolution (YOLOv8n FP32)')
        fig.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'), dpi=150)
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'), dpi=150)
        plt.close()

    # Plot 2: Latency & FPS vs Model Size (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'].str.upper() == 'FP32')]
    if not model_df.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:red'
        ax1.set_xlabel('Model Size')
        ax1.set_ylabel('Average Latency (ms)', color=color)
        bars1 = ax1.bar(model_df['Model'], model_df['Average_Latency_ms'], color=color, alpha=0.6, width=0.4, label='Latency (ms)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.bar_label(bars1, fmt='%.1f ms', padding=3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Average FPS', color=color)
        bars2 = ax2.bar(model_df['Model'], model_df['Average_FPS'], color=color, alpha=0.6, width=0.2, label='FPS')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.bar_label(bars2, fmt='%.1f FPS', padding=3)

        plt.title('Performance Comparison by Model Size (640x640 FP32)')
        fig.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'), dpi=150)
        plt.close()

    # Plot 3: Latency & FPS vs Precision (YOLOv8n, 640x640)
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'].str.lower() == 'yolov8n')]
    if not prec_df.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:red'
        ax1.set_xlabel('Precision Format')
        ax1.set_ylabel('Average Latency (ms)', color=color)
        bars1 = ax1.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=color, alpha=0.6, width=0.4, label='Latency (ms)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.bar_label(bars1, fmt='%.1f ms', padding=3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Average FPS', color=color)
        bars2 = ax2.bar(prec_df['Precision'], prec_df['Average_FPS'], color=color, alpha=0.6, width=0.2, label='FPS')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.bar_label(bars2, fmt='%.1f FPS', padding=3)

        plt.title('Performance Comparison by Precision Level (YOLOv8n 640x640)')
        fig.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_precision.png'), dpi=150)
        plt.close()

    print(f"All plots saved successfully in {plots_dir}")

if __name__ == '__main__':
    main()
