import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Plot 1: Average Latency vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(8, 5))
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    plt.savefig('latency_vs_resolution.png')
plt.close()

# Plot 2: Average FPS vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(8, 5))
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig('fps_vs_resolution.png')
plt.close()

# Plot 3: Precision Comparison (FP32 vs FP16 at 640x640)
plt.figure(figsize=(8, 5))
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if not prec_df.empty:
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    plt.savefig('latency_precision_comp.png')
plt.close()

# Plot 4: Model Size Comparison (Nano vs Small at 640x640, FP32)
plt.figure(figsize=(8, 5))
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['gold', 'lightcoral'])
    plt.title('FPS vs Model Variant (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig('fps_vs_model.png')
plt.close()

print('Enhanced plots saved successfully in results/plots')
