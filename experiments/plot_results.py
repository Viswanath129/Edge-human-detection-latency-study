import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Determine directories using absolute paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    csv_path = os.path.join(project_root, 'results', 'tables', 'summary.csv')
    plots_dir = os.path.join(project_root, 'results', 'plots')

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Summary CSV not found at {csv_path}. Cannot generate plots.")
        return

    # Load the summary data
    df = pd.read_csv(csv_path)
    print("Loaded summary data:")
    print(df)

    # 1. Plot 1: Average Latency vs Resolution (for FP32 models like YOLOv8n)
    # Let's filter or plot the resolution comparison if present
    res_df = df[df['Model'].str.lower().str.startswith('yolov8n') & (df['Precision'].str.upper() == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + (max(res_df['Average_Latency_ms']) * 0.02), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'), dpi=300)
        plt.close()

        # Plot 2: Average FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + (max(res_df['Average_FPS']) * 0.02), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'), dpi=300)
        plt.close()

    # 3. Plot 3: Model Size Comparison (e.g., YOLOv8n vs YOLOv8s at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'].str.upper() == 'FP32')]
    if len(model_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Model Size (640x640 FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(model_df['Average_Latency_ms']):
            plt.text(i, v + (max(model_df['Average_Latency_ms']) * 0.02), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Model Size (640x640 FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + (max(model_df['Average_FPS']) * 0.02), f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fps_vs_model.png'), dpi=300)
        plt.close()

    print(f"Plots saved successfully in {plots_dir}")

if __name__ == '__main__':
    main()
