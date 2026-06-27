import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(script_dir, "../tables/summary.csv")
output_dir = script_dir

if not os.path.exists(summary_path):
    print(f"Error: {summary_path} not found.")
    exit(1)

df = pd.read_csv(summary_path)

# Ensure numeric types
df['Average_Latency_ms'] = pd.to_numeric(df['Average_Latency_ms'])
df['Average_FPS'] = pd.to_numeric(df['Average_FPS'])

# 1. Latency vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(10, 6))
subset = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not subset.empty:
    plt.bar(subset['Resolution'], subset['Average_Latency_ms'], color='salmon')
    plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(subset['Average_Latency_ms']):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'latency_vs_resolution.png'))
plt.close()

# 2. FPS vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(10, 6))
if not subset.empty:
    plt.bar(subset['Resolution'], subset['Average_FPS'], color='lightgreen')
    plt.title('Throughput (FPS) vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('FPS')
    for i, v in enumerate(subset['Average_FPS']):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'fps_vs_resolution.png'))
plt.close()

# 3. Latency vs Model Size (at 640x640, FP32)
plt.figure(figsize=(10, 6))
subset_model = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not subset_model.empty:
    plt.bar(subset_model['Model'], subset_model['Average_Latency_ms'], color='skyblue')
    plt.title('Inference Latency vs Model Variant (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(subset_model['Average_Latency_ms']):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'latency_vs_model.png'))
plt.close()

# 4. Precision Comparison (if multiple precisions exist)
if df['Precision'].nunique() > 1:
    plt.figure(figsize=(10, 6))
    subset_prec = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
    plt.bar(subset_prec['Precision'], subset_prec['Average_Latency_ms'], color='orchid')
    plt.title('Precision Impact on Latency (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(subset_prec['Average_Latency_ms']):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'latency_precision_comp.png'))
plt.close()

print('Enhanced plots generated in results/plots/')
