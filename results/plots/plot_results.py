import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
csv_path = '../tables/summary.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# 1. Latency vs Resolution (Standardized on YOLOv8n, FP32)
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Resolution (Standardized on YOLOv8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Inference FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# 3. Latency: FP32 vs FP16 (Standardized on YOLOv8n, 640x640)
prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')].sort_values('Precision')
if len(prec_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Precision Impact on Latency (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
    plt.savefig('latency_precision_comp.png')
    plt.close()

# 4. FPS vs Model Size (Standardized on 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')].sort_values('Average_FPS', ascending=False)
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['gold', 'slateblue'])
    plt.title('Inference FPS vs Model Complexity (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + (v * 0.02), f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_model.png')
    plt.close()

print('Plots generated successfully in results/plots/')
