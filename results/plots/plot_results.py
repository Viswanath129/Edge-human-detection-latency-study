import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    # Path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "..", "tables", "summary.csv")

    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    os.chdir(script_dir)

    # 1. Latency vs Resolution (for yolov8n, FP32)
    res_df = df[(df["Model"] == "yolov8n") & (df["Precision"] == "FP32")].sort_values("Resolution", ascending=False)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df["Resolution"], res_df["Average_Latency_ms"], color='salmon')
        plt.title('Latency vs Input Resolution (yolov8n, FP32)')
        plt.ylabel('Avg Latency (ms)')
        plt.savefig('latency_vs_resolution.png')
        plt.close()

    # 2. FPS vs Model Size (for 640x640, FP32)
    model_df = df[(df["Resolution"] == "640x640") & (df["Precision"] == "FP32")]
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df["Model"], model_df["Average_FPS"], color='lightgreen')
        plt.title('FPS vs Model Variant (640x640, FP32)')
        plt.ylabel('Avg FPS')
        plt.savefig('fps_vs_model.png')
        plt.close()

    # 3. Precision Comparison (yolov8n, 640x640)
    prec_df = df[(df["Model"] == "yolov8n") & (df["Resolution"] == "640x640")]
    if not prec_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df["Precision"], prec_df["Average_Latency_ms"], color='lightblue')
        plt.title('Latency Comparison: FP32 vs FP16 (yolov8n, 640x640)')
        plt.ylabel('Avg Latency (ms)')
        plt.savefig('latency_precision_comp.png')
        plt.close()

    print('Plots updated successfully in results/plots/')

if __name__ == "__main__":
    plot_results()
