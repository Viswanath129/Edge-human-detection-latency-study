import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory to the script's directory for relative path consistency
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the summary data
csv_path = '../tables/summary.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

def plot_bar(df, x_col, y_col, title, ylabel, filename, color='skyblue'):
    plt.figure(figsize=(10, 6))

    # For categorical combinations
    if x_col == 'Combined':
        df['Label'] = df['Resolution'] + '\n' + df['Model'] + '\n' + df['Precision']
        x_data = df['Label']
    else:
        x_data = df[x_col]

    bars = plt.bar(x_data, df[y_col], color=color)
    plt.title(title)
    plt.xlabel('Experiment Configuration')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), round(yval, 2), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

# Plot 1: Latency vs Resolution (for yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')].sort_values('Resolution', ascending=False)
if not res_df.empty:
    plot_bar(res_df, 'Resolution', 'Average_Latency_ms', 'Latency vs Input Resolution (yolov8n, FP32)', 'Avg Latency (ms)', 'latency_vs_resolution.png', 'salmon')
    plot_bar(res_df, 'Resolution', 'Average_FPS', 'FPS vs Input Resolution (yolov8n, FP32)', 'Avg FPS', 'fps_vs_resolution.png', 'lightgreen')

# Plot 2: Model Size comparison (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plot_bar(model_df, 'Model', 'Average_Latency_ms', 'Latency vs Model Size (640x640, FP32)', 'Avg Latency (ms)', 'latency_vs_model.png', 'lightblue')
    plot_bar(model_df, 'Model', 'Average_FPS', 'FPS vs Model Size (640x640, FP32)', 'Avg FPS', 'fps_vs_model.png', 'orange')

# Plot 3: Precision comparison (if FP16 exists)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
if len(prec_df) > 1:
    plot_bar(prec_df, 'Precision', 'Average_Latency_ms', 'Latency vs Precision (yolov8n, 640x640)', 'Avg Latency (ms)', 'latency_vs_precision.png', 'violet')

print('All plots generated successfully in results/plots')
