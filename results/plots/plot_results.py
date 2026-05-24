import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data using absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.abspath(os.path.join(script_dir, "..", "tables", "summary.csv"))

if not os.path.exists(summary_path):
    print(f"Summary file not found at {summary_path}. Please run benchmarks first.")
    exit(1)

df = pd.read_csv(summary_path)

# Create plots directory if it doesn't exist
plots_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(plots_dir, exist_ok=True)

def plot_metric(data, x_col, y_col, title, filename, color_map=None):
    plt.figure(figsize=(10, 6))
    colors = [color_map.get(x, 'skyblue') for x in data[x_col]] if color_map else 'skyblue'
    bars = plt.bar(data[x_col], data[y_col], color=colors)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

# 1. Resolution Comparison (yolov8n.pt, FP32)
res_df = df[(df['Model'] == 'yolov8n.pt') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plot_metric(res_df, 'Resolution', 'Average_FPS', 'FPS vs Resolution (yolov8n, FP32)', 'fps_vs_resolution.png',
                {'640x640': 'salmon', '416x416': 'lightblue'})
    plot_metric(res_df, 'Resolution', 'Average_Latency_ms', 'Latency vs Resolution (yolov8n, FP32)', 'latency_vs_resolution.png',
                {'640x640': 'salmon', '416x416': 'lightblue'})

# 2. Model Size Comparison (640x640, FP32)
size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')].sort_values('Model')
if not size_df.empty:
    plot_metric(size_df, 'Model', 'Average_FPS', 'FPS vs Model Size (640x640, FP32)', 'fps_vs_model.png')
    plot_metric(size_df, 'Model', 'Average_Latency_ms', 'Latency vs Model Size (640x640, FP32)', 'latency_vs_model.png')

# 3. Precision Comparison (yolov8n.pt, 640x640)
prec_df = df[(df['Model'] == 'yolov8n.pt') & (df['Resolution'] == '640x640')].sort_values('Precision')
if not prec_df.empty:
    plot_metric(prec_df, 'Precision', 'Average_FPS', 'FPS vs Precision (yolov8n, 640x640)', 'fps_vs_precision.png')
    plot_metric(prec_df, 'Precision', 'Average_Latency_ms', 'Latency vs Precision (yolov8n, 640x640)', 'latency_vs_precision.png')

print(f"All plots generated successfully in {plots_dir}")
