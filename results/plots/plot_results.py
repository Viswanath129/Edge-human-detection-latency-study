import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# 1. Latency vs Resolution (for yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
    plt.title('Inference Latency vs Resolution (yolov8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Resolution (for yolov8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
    plt.title('FPS vs Resolution (yolov8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# 3. Latency: FP32 vs FP16 (at 640x640, yolov8n)
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if len(prec_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='lightblue')
    plt.title('Precision Comparison: FP32 vs FP16 (640x640, yolov8n)')
    plt.xlabel('Precision')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_precision_comp.png')
    plt.close()

# 4. FPS: Nano vs Small (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if len(model_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color='orange')
    plt.title('Model Size Comparison: Nano vs Small (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_model.png')
    plt.close()

print('Enhanced plots saved successfully in results/plots')
