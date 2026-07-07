import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Ensure directory exists
os.makedirs('.', exist_ok=True)

def plot_bar(subset, x_col, y_col, title, filename, color):
    plt.figure(figsize=(10, 6))
    plt.bar(subset[x_col], subset[y_col], color=color)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col.replace('_', ' '))
    for i, v in enumerate(subset[y_col]):
        plt.text(i, v + (v * 0.02), str(v), ha='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Resolution Comparison (yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plot_bar(res_df, 'Resolution', 'Average_Latency_ms', 'Latency vs Resolution (yolov8n, FP32)', 'latency_vs_resolution.png', 'salmon')
    plot_bar(res_df, 'Resolution', 'Average_FPS', 'FPS vs Resolution (yolov8n, FP32)', 'fps_vs_resolution.png', 'lightgreen')

# 2. Model Size Comparison (640x640, FP32)
size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not size_df.empty:
    plot_bar(size_df, 'Model', 'Average_Latency_ms', 'Latency vs Model Size (640x640, FP32)', 'latency_vs_model.png', 'skyblue')
    plot_bar(size_df, 'Model', 'Average_FPS', 'FPS vs Model Size (640x640, FP32)', 'fps_vs_model.png', 'orange')

# 3. Precision Comparison (yolov8n, 640x640)
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if not prec_df.empty and len(prec_df) > 1:
    plot_bar(prec_df, 'Precision', 'Average_Latency_ms', 'Latency vs Precision (yolov8n, 640x640)', 'latency_vs_precision.png', 'plum')
    plot_bar(prec_df, 'Precision', 'Average_FPS', 'FPS vs Precision (yolov8n, 640x640)', 'fps_vs_precision.png', 'gold')

print('Plots updated successfully in results/plots')
