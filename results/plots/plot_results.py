import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../tables/summary.csv')

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Create plots directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename))
    plt.close()
    print(f"Saved {filename}")

# 1. Latency vs Resolution (for yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color='salmon')
    plt.title('Inference Latency vs Resolution (yolov8n, FP32)')
    plt.ylabel('Latency (ms)')
    save_plot('latency_vs_resolution.png')

# 2. FPS vs Resolution (for yolov8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='lightgreen')
    plt.title('FPS vs Resolution (yolov8n, FP32)')
    plt.ylabel('FPS')
    save_plot('fps_vs_resolution.png')

# 3. Latency vs Model Size (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')].sort_values('Average_Latency_ms')
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_Latency_ms'], color='lightblue')
    plt.title('Inference Latency vs Model Size (640x640, FP32)')
    plt.ylabel('Latency (ms)')
    save_plot('latency_vs_model.png')

# 4. FPS vs Model Size (at 640x640, FP32)
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color='orange')
    plt.title('FPS vs Model Size (640x640, FP32)')
    plt.ylabel('FPS')
    save_plot('fps_vs_model.png')

print('Plotting completed.')
