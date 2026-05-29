import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

def plot_bar(df, x, y, title, xlabel, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    plt.bar(df[x], df[y], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    for i, v in enumerate(df[y]):
        plt.text(i, v + (v * 0.02), str(v), ha='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Resolution Comparison (YOLOv8n, FP32)
res_df = df[(df['Model'] == 'YOLOv8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
plot_bar(res_df, 'Resolution', 'Average_FPS', 'FPS vs Resolution (YOLOv8n, FP32)', 'Resolution', 'FPS', 'fps_vs_resolution.png', 'lightgreen')
plot_bar(res_df, 'Resolution', 'Average_Latency_ms', 'Latency vs Resolution (YOLOv8n, FP32)', 'Resolution', 'Latency (ms)', 'latency_vs_resolution.png', 'salmon')

# 2. Model Size Comparison (640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
plot_bar(model_df, 'Model', 'Average_FPS', 'FPS vs Model Size (640x640, FP32)', 'Model', 'FPS', 'fps_vs_model.png', 'skyblue')
plot_bar(model_df, 'Model', 'Average_Latency_ms', 'Latency vs Model Size (640x640, FP32)', 'Model', 'Latency (ms)', 'latency_vs_model.png', 'orchid')

# 3. Precision Comparison (640x640, YOLOv8n)
# Note: FP16 might be very slow on CPU, so we might want to log scale or just show it
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'YOLOv8n')]
plot_bar(prec_df, 'Precision', 'Average_FPS', 'FPS vs Precision (640x640, YOLOv8n)', 'Precision', 'FPS', 'fps_vs_precision.png', 'gold')

print('Plots updated successfully in results/plots')
