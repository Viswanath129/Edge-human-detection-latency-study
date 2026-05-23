import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
summary_path = os.path.join(base_dir, "tables", "summary.csv")
df = pd.read_csv(summary_path)

# Create plots directory if it doesn't exist
plots_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(plots_dir, exist_ok=True)

def create_bar_plot(data, x_col, y_col, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_col], data[y_col], color=color)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    for i, v in enumerate(data[y_col]):
        plt.text(i, v + (v * 0.02), str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

# 1. Resolution Comparison (yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
if not res_df.empty:
    create_bar_plot(res_df, 'Resolution', 'Average_Latency_ms', 'Latency vs Resolution (yolov8n, FP32)', 'Latency (ms)', 'latency_vs_resolution.png', 'salmon')
    create_bar_plot(res_df, 'Resolution', 'Average_FPS', 'FPS vs Resolution (yolov8n, FP32)', 'FPS', 'fps_vs_resolution.png', 'lightgreen')

# 2. Precision Comparison (yolov8n, 640x640)
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if not prec_df.empty:
    create_bar_plot(prec_df, 'Precision', 'Average_Latency_ms', 'Latency vs Precision (yolov8n, 640x640)', 'Latency (ms)', 'latency_vs_precision.png', 'lightblue')

# 3. Model Size Comparison (640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    create_bar_plot(model_df, 'Model', 'Average_Latency_ms', 'Latency vs Model Size (640x640, FP32)', 'Latency (ms)', 'latency_vs_model.png', 'orange')

print(f'Plots saved successfully in {plots_dir}')
