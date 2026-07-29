import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Resolve absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, '../results/tables/summary.csv'))
    plot_dir = os.path.abspath(os.path.join(current_dir, '../results/plots'))

    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Summary CSV not found at {summary_path}. Skipping plot generation.")
        return

    # Load summary data
    df = pd.read_csv(summary_path)
    if df.empty:
        print("Summary CSV is empty. Skipping plot generation.")
        return

    # Normalize/clean columns to avoid issues
    df['Resolution'] = df['Resolution'].astype(str)
    df['Model'] = df['Model'].astype(str)
    df['Precision'] = df['Precision'].astype(str)
    df['Average_FPS'] = df['Average_FPS'].astype(float)
    df['Average_Latency_ms'] = df['Average_Latency_ms'].astype(float)

    # Plot 1: Latency vs Resolution (for YOLOv8n & FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        res_df = res_df.sort_values(by='Resolution', ascending=False)  # 640 before 416
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'][:len(res_df)])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'latency_vs_resolution.png'))
        plt.close()

        # Also FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'][:len(res_df)])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'fps_vs_resolution.png'))
        plt.close()

    # Plot 2: Latency vs Model Size (for 640x640 & FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        model_df = model_df.sort_values(by='Model')
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color=['salmon', 'lightblue'][:len(model_df)])
        plt.title('Average Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'latency_vs_model.png'))
        plt.close()

    print(f"Plots successfully generated and saved in {plot_dir}")

if __name__ == '__main__':
    main()
