import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
summary_path = '../tables/summary.csv'
if not os.path.exists(summary_path):
    print(f"Error: {summary_path} not found. Run benchmarks first.")
    exit(1)

df = pd.read_csv(summary_path)

# Ensure plots directory exists
os.makedirs('.', exist_ok=True)

# 1. Latency vs Resolution (for yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
    plt.title('Inference Latency vs Resolution (yolov8n, FP32)')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Resolution (for yolov8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
    plt.title('FPS vs Resolution (yolov8n, FP32)')
    plt.ylabel('FPS')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# 3. FPS vs Model Size (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color='skyblue')
    plt.title('FPS vs Model Size (640x640, FP32)')
    plt.ylabel('FPS')
    plt.savefig('fps_vs_model.png')
    plt.close()

# 4. Latency: FP32 vs FP16 (at 640x640, yolov8n)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
if len(prec_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='orange')
    plt.title('Latency Comparison: FP32 vs FP16 (640x640, yolov8n)')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_precision_comp.png')
    plt.close()

print('Plots updated successfully in results/plots/')
