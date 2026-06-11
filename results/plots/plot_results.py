import pandas as pd
import matplotlib.pyplot as plt
import os

# Set absolute paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "../tables")
SUMMARY_CSV = os.path.join(TABLES_DIR, "summary.csv")

def generate_plots():
    if not os.path.exists(SUMMARY_CSV):
        print(f"Summary file not found: {SUMMARY_CSV}")
        return

    df = pd.read_csv(SUMMARY_CSV)
    os.chdir(SCRIPT_DIR)

    # 1. Latency vs Resolution (for yolov8n, FP32)
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 2, str(v), ha='center')
        plt.savefig('latency_vs_resolution.png')
        plt.close()

        # 2. FPS vs Resolution
        plt.figure(figsize=(8, 5))
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Resolution')
        plt.ylabel('FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('fps_vs_resolution.png')
        plt.close()

    # 3. Latency/Precision Comparison (for yolov8n at 640x640)
    prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
    if len(prec_df) > 1:
        plt.figure(figsize=(8, 5))
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='lightblue')
        plt.title('Inference Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
        plt.xlabel('Precision')
        plt.ylabel('Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 2, str(v), ha='center')
        plt.savefig('latency_precision_comp.png')
        plt.close()

    # 4. FPS vs Model Size (at 640x640, FP32)
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(model_df['Model'], model_df['Average_FPS'], color='orange')
        plt.title('Average FPS vs Model Variant (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('FPS')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('fps_vs_model.png')
        plt.close()

    print(f"Plots updated in {SCRIPT_DIR}")

if __name__ == "__main__":
    generate_plots()
