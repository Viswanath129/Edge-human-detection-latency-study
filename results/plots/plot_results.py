import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
table_path = os.path.join(os.path.dirname(__file__), '../tables/summary.csv')
if not os.path.exists(table_path):
    print(f"Summary table not found at {table_path}. Please run benchmarks first.")
    exit(1)

df = pd.read_csv(table_path)

# Ensure plots directory exists
os.makedirs(os.path.dirname(__file__), exist_ok=True)

def plot_comparison(subset, x_col, title, filename, ylabel='Average Latency (ms)', y_col='Average_Latency_ms'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(subset[x_col], subset[y_col], color='skyblue')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(subset[y_col]) * 0.02), round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))
    plt.close()
    print(f"Saved {filename}")

# 1. Resolution Comparison (yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plot_comparison(res_df, 'Resolution', 'Inference Latency vs Resolution (yolov8n, FP32)', 'latency_vs_resolution.png')
    plot_comparison(res_df, 'Resolution', 'Inference FPS vs Resolution (yolov8n, FP32)', 'fps_vs_resolution.png', ylabel='FPS', y_col='Average_FPS')

# 2. Precision Comparison (640x640, yolov8n)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
if not prec_df.empty:
    plot_comparison(prec_df, 'Precision', 'Inference Latency vs Precision (640x640, yolov8n)', 'latency_vs_precision.png')

# 3. Model Size Comparison (640x640, FP32)
size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not size_df.empty:
    plot_comparison(size_df, 'Model', 'Inference Latency vs Model Size (640x640, FP32)', 'latency_vs_model_size.png')

print('Visualization updates complete.')
