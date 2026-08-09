import os
import pandas as pd
import matplotlib.pyplot as plt

# Get absolute paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.abspath(os.path.join(current_dir, "../results/tables/summary.csv"))
plots_dir = os.path.abspath(os.path.join(current_dir, "../results/plots"))

os.makedirs(plots_dir, exist_ok=True)

def run_plotting():
    if not os.path.exists(summary_path):
        print(f"Summary table not found at {summary_path}. Cannot plot.")
        return

    df = pd.read_csv(summary_path)

    # Normalize/ensure data types
    df['Average_Latency_ms'] = pd.to_numeric(df['Average_Latency_ms'], errors='coerce')
    df['Average_FPS'] = pd.to_numeric(df['Average_FPS'], errors='coerce')
    df['Resolution'] = df['Resolution'].astype(str)
    df['Model'] = df['Model'].astype(str)
    df['Precision'] = df['Precision'].astype(str)

    # Plot 1: Latency vs Resolution (for FP32 YOLOv8n)
    df_res = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
    if not df_res.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_res['Average_Latency_ms']):
            if pd.notna(v):
                plt.text(i, v + 2, f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(df_res['Average_FPS']):
            if pd.notna(v):
                plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
        plt.close()

    # Plot 2: Latency vs Model Size (for 640x640, FP32)
    df_size = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not df_size.empty:
        model_order = {'YOLOv8n': 1, 'YOLOv8s': 2, 'YOLOv8m': 3}
        df_size = df_size.copy()
        df_size['order'] = df_size['Model'].map(model_order).fillna(99)
        df_size = df_size.sort_values('order')

        plt.figure(figsize=(8, 5))
        plt.bar(df_size['Model'], df_size['Average_Latency_ms'], color='skyblue')
        plt.title('Average Latency vs Model Size (640x640, FP32)')
        plt.xlabel('Model Architecture')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_size['Average_Latency_ms']):
            if pd.notna(v):
                plt.text(i, v + 2, f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_model.png'))
        plt.close()

    # Plot 3: Precision comparison (FP32 vs FP16 for YOLOv8n at 640x640)
    df_prec = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
    if len(df_prec) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(df_prec['Precision'], df_prec['Average_Latency_ms'], color=['lightcoral', 'lightgreen'])
        plt.title('Average Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(df_prec['Average_Latency_ms']):
            if pd.notna(v):
                plt.text(i, v + 2, f"{v:.1f}", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'latency_vs_precision.png'))
        plt.close()

    print('Plots saved successfully in results/plots')

if __name__ == "__main__":
    run_plotting()
