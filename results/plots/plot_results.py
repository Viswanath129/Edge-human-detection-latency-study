import pandas as pd
import matplotlib.pyplot as plt
import os

# Use absolute paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(BASE_DIR, "..", "tables", "summary.csv")
PLOT_DIR = BASE_DIR

# Load the summary data
if not os.path.exists(TABLE_PATH):
    print(f"Error: {TABLE_PATH} not found.")
    exit(1)

df = pd.read_csv(TABLE_PATH)

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(filename):
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    print(f"Saved plot to {path}")
    plt.close()

# Plot 1: Average Latency vs Resolution (for yolov8n, FP32)
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 2, str(v), ha='center')
    save_plot('latency_vs_resolution.png')

# Plot 2: Average FPS vs Resolution (for yolov8n, FP32)
if not res_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Input Resolution')
    plt.ylabel('Average FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.5, str(v), ha='center')
    save_plot('fps_vs_resolution.png')

# Plot 3: FPS vs Model (at 640x640, FP32)
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['teal', 'orchid'])
    plt.title('Average FPS vs Model Size (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('Average FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.2, str(v), ha='center')
    save_plot('fps_vs_model.png')

print('All plots generated successfully.')
