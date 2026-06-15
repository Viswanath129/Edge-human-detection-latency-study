import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# 1. Latency vs Resolution (Filter for YOLOv8n, FP32)
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Resolution
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# 3. FPS vs Model Size (Filter for 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['skyblue', 'plum'])
    plt.title('Average FPS vs Model Size (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_model.png')
    plt.close()

# 4. Latency Precision Comparison (Filter for YOLOv8n, 640x640)
prec_df = df[(df['Model'] == 'YOLOv8n') & (df['Resolution'] == '640x640')]
if not prec_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['grey', 'gold'])
    plt.title('Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_precision_comp.png')
    plt.close()

print('Plots saved successfully in results/plots')
