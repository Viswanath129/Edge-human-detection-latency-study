import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data using robust path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../tables/summary.csv')

if not os.path.exists(csv_path):
    print(f"Error: Summary file not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Ensure plots directory exists (where the script is located)
os.makedirs(script_dir, exist_ok=True)

def save_plot(filename):
    path = os.path.join(script_dir, filename)
    plt.savefig(path)
    print(f"Saved plot to {path}")

# 1. Latency vs Resolution (YOLOv8n, FP32)
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Inference Latency vs Input Resolution (YOLOv8n)')
    plt.xlabel('Resolution')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    save_plot('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Model Size (640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Throughput (FPS) vs Model Size (640x640)')
    plt.xlabel('Model Variant')
    plt.ylabel('FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.2, f"{v:.1f}", ha='center')
    save_plot('fps_vs_model.png')
    plt.close()

# 3. Precision Comparison (YOLOv8n, 640x640)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')]
if not prec_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['plum', 'skyblue'])
    plt.title('Precision Impact on Latency (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    save_plot('latency_precision_comp.png')
    plt.close()
