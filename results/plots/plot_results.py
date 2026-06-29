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

# Plot 1: Average Latency vs Resolution (for YOLOv8n, FP32)
res_df = df[(df['Model'] == 'yolov8n.pt') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

# Plot 2: Average FPS vs Resolution (for YOLOv8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_resolution.png')
    plt.close()

# Plot 3: Latency Comparison - FP32 vs FP16 (for YOLOv8n, 640x640)
prec_df = df[(df['Model'] == 'yolov8n.pt') & (df['Resolution'] == '640x640')]
if len(prec_df) > 1:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['teal', 'orchid'])
    plt.title('Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 2, f"{v:.1f}", ha='center')
    plt.savefig('latency_precision_comp.png')
    plt.close()

# Plot 4: FPS vs Model Size (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['gold', 'skyblue'])
    plt.title('FPS vs Model Architecture (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig('fps_vs_model.png')
    plt.close()

print('Enhanced plots saved successfully in results/plots')
