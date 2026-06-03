import pandas as pd
import matplotlib.pyplot as plt
import os

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../tables/summary.csv')
plots_dir = base_dir

# Load the summary data
df = pd.read_csv(csv_path)

def save_plot(filename, title, x, y, hue_col=None):
    plt.figure(figsize=(10, 6))

    if hue_col:
        # Grouped bar chart
        pivot_df = df.pivot_table(index=x, columns=hue_col, values=y)
        pivot_df.plot(kind='bar', ax=plt.gca())
    else:
        plt.bar(df[x], df[y], color='skyblue')

    plt.title(title)
    plt.xlabel(x.replace('_', ' '))
    plt.ylabel(y.replace('_', ' '))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

# 1. Resolution Comparison (yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
plt.figure(figsize=(8, 5))
plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
plt.title('FPS vs Resolution (yolov8n, FP32)')
plt.ylabel('Average FPS')
plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
plt.close()

# 2. Model Size Comparison (640x640, FP32)
size_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
plt.figure(figsize=(8, 5))
plt.bar(size_df['Model'], size_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
plt.title('Latency vs Model Size (640x640, FP32)')
plt.ylabel('Average Latency (ms)')
plt.savefig(os.path.join(plots_dir, 'latency_vs_model_size.png'))
plt.close()

# 3. Precision Comparison (yolov8n, 640x640)
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
plt.figure(figsize=(8, 5))
plt.bar(prec_df['Precision'], prec_df['Average_FPS'], color=['orchid', 'gold'])
plt.title('FPS vs Precision (yolov8n, 640x640)')
plt.ylabel('Average FPS')
plt.savefig(os.path.join(plots_dir, 'fps_vs_precision.png'))
plt.close()

print('Plots generated successfully in results/plots')
