import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# 1. Latency vs Resolution (for yolov8n, FP32)
df_res = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not df_res.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(df_res['Resolution'], df_res['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Inference Latency vs Resolution (YOLOv8n, FP32)')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# 2. FPS vs Resolution (for yolov8n, FP32)
if not df_res.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(df_res['Resolution'], df_res['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('FPS vs Resolution (YOLOv8n, FP32)')
    plt.ylabel('FPS')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# 3. Model Comparison (640x640, FP32)
df_model = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not df_model.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(df_model['Model'], df_model['Average_FPS'], color=['skyblue', 'plum'])
    plt.title('FPS Comparison: YOLOv8n vs YOLOv8s (640x640)')
    plt.ylabel('FPS')
    plt.savefig('fps_vs_model.png')
    plt.close()

# 4. Precision Comparison (YOLOv8n, 640x640)
df_prec = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if len(df_prec) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(df_prec['Precision'], df_prec['Average_Latency_ms'], color=['gold', 'lightcoral'])
    plt.title('Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_precision_comp.png')
    plt.close()

print('Plots saved successfully in results/plots')
