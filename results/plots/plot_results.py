import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
summary_path = '../tables/summary.csv'
if not os.path.exists(summary_path):
    print(f"Error: {summary_path} not found.")
    exit(1)

df = pd.read_csv(summary_path)

# Plot 1: Average Latency vs Resolution (for YOLOv8n, FP32)
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# Plot 2: Latency vs Model Size (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')].sort_values('Average_Latency_ms')
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Latency vs Model Size (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(model_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    plt.savefig('latency_vs_model.png')
    plt.close()

# Plot 3: Latency vs Precision (at 640x640, YOLOv8n)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')].sort_values('Precision')
if len(prec_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['gold', 'skyblue'])
    plt.title('Latency vs Precision (640x640, YOLOv8n)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    plt.savefig('latency_vs_precision.png')
    plt.close()

print('Plots updated successfully in results/plots')
