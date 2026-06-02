import pandas as pd
import matplotlib.pyplot as plt
import os

# Use absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(script_dir, '../tables/summary.csv')
output_dir = script_dir

# Load the summary data
df = pd.read_csv(summary_path)

def save_plot(name):
    path = os.path.join(output_dir, name)
    plt.savefig(path)
    print(f"Saved plot: {path}")
    plt.close()

# Plot 1: Resolution Impact (YOLOv8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color='skyblue')
    plt.title('FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('Average FPS')
    save_plot('fps_vs_resolution.png')

# Plot 2: Model Size Impact (640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color='salmon')
    plt.title('FPS vs Model Size (640x640, FP32)')
    plt.xlabel('Model')
    plt.ylabel('Average FPS')
    save_plot('fps_vs_model.png')

# Plot 3: Precision Impact (640x640, YOLOv8n)
prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
if not prec_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(prec_df['Precision'], prec_df['Average_FPS'], color='lightgreen')
    plt.title('FPS vs Precision (640x640, YOLOv8n)')
    plt.xlabel('Precision')
    plt.ylabel('Average FPS')
    save_plot('fps_vs_precision.png')
