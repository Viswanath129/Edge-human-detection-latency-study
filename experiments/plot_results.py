import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("--- Generating Performance Plots ---")

    # Locate summary.csv and plots directory using absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(base_dir, "..", "results", "tables", "summary.csv"))
    plots_dir = os.path.abspath(os.path.join(base_dir, "..", "results", "plots"))
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}. Skipping plotting.")
        return

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return

    # 1. Latency vs Resolution (for YOLOv8n, FP32)
    df_res = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not df_res.empty:
        # Sort by resolution (416 before 640)
        df_res = df_res.sort_values(by='Resolution')

        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['salmon', 'lightblue'][:len(df_res)])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_res['Average_Latency_ms']):
            plt.text(i, v + (0.02 * max(df_res['Average_Latency_ms'])), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

        # FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['lightgreen', 'orange'][:len(df_res)])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(df_res['Average_FPS']):
            plt.text(i, v + (0.02 * max(df_res['Average_FPS'])), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
        plt.close()
        print("Generated resolution comparison plots.")

    # 2. Latency vs Model Size (for 640x640, FP32)
    df_model = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not df_model.empty:
        # Sort model sizes: YOLOv8n first, then YOLOv8s
        df_model = df_model.sort_values(by='Model', ascending=False)

        plt.figure(figsize=(8, 5))
        plt.bar(df_model['Model'], df_model['Average_Latency_ms'], color=['salmon', 'lightblue'][:len(df_model)])
        plt.title('Average Latency vs Model Variant (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_model['Average_Latency_ms']):
            plt.text(i, v + (0.02 * max(df_model['Average_Latency_ms'])), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
        plt.close()
        print("Generated model size comparison plots.")

    print("All plots generated successfully in results/plots.")

if __name__ == "__main__":
    main()
