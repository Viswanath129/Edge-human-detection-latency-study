import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("Generating Comparative Visualizations...")

    # Deriving absolute paths to find summary.csv and save plots correctly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.abspath(os.path.join(current_dir, "../results/tables/summary.csv"))
    plots_dir = os.path.abspath(os.path.join(current_dir, "../results/plots"))

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(summary_path):
        print(f"Error: summary.csv not found at {summary_path}. Run experiments first.")
        return

    df = pd.read_csv(summary_path)
    print(f"Loaded summary with columns: {list(df.columns)}")

    # 1. Latency & FPS vs Resolution (YOLOv8n, FP32)
    res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]

    if not res_df.empty:
        # Latency vs Resolution Bar Plot
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'], width=0.4)
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 1, f"{v:.2f}", ha='center', fontweight='bold')
        out_path = os.path.join(plots_dir, 'latency_vs_resolution.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

        # FPS vs Resolution Bar Plot
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'], width=0.4)
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.2, f"{v:.2f}", ha='center', fontweight='bold')
        out_path = os.path.join(plots_dir, 'fps_vs_resolution.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # 2. Latency: FP32 vs FP16 (YOLOv8n, 640x640) - only plotted if FP16 is evaluated (CUDA environments)
    prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
    if len(prec_df) >= 2:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['skyblue', 'plum'], width=0.4)
        plt.title('Latency Comparison: FP32 vs FP16 (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 2, f"{v:.2f}", ha='center', fontweight='bold')
        out_path = os.path.join(plots_dir, 'latency_precision_comp.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # 3. FPS Comparison: Model Sizes (640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32') & (df['Model'].isin(['YOLOv8n', 'YOLOv8s']))]
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color=['lightcoral', 'lightseagreen'], width=0.4)
        plt.title('FPS Comparison: YOLOv8 Nano vs Small (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average FPS (Frames/Sec)')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
        out_path = os.path.join(plots_dir, 'fps_vs_model.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
