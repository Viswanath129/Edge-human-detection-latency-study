import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
# The script is expected to be run from results/plots/
df = pd.read_csv('../tables/summary.csv')

# Plot 1: Average Latency vs Resolution (for YOLOv8n, FP32)
plt.figure(figsize=(10, 6))
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='skyblue')
    plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig('latency_vs_resolution.png')
plt.close()

# Plot 2: Average FPS vs Resolution (for YOLOv8n, FP32)
plt.figure(figsize=(10, 6))
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
    plt.title('Inference FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.savefig('fps_vs_resolution.png')
plt.close()

# Plot 3: FPS vs Model Size (at 640x640, FP32)
plt.figure(figsize=(10, 6))
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.bar(model_df['Model'], model_df['Average_FPS'], color='salmon')
    plt.title('Inference FPS vs Model Variant (640x640, FP32)')
    plt.xlabel('Model')
    plt.ylabel('FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.savefig('fps_vs_model.png')
plt.close()

# Plot 4: Latency Comparison (Precision)
plt.figure(figsize=(10, 6))
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')]
if not prec_df.empty and len(prec_df) > 1:
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color='plum')
    plt.title('Latency Comparison: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig('latency_precision_comp.png')
plt.close()

print('Enhanced plots saved successfully in results/plots')
