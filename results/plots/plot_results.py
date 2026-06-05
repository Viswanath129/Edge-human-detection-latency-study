import pandas as pd
import matplotlib.pyplot as plt
import os

# Get absolute paths relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../tables/summary.csv')
plots_dir = script_dir

# Load the summary data
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# 1. Latency vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(10, 6))
res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32') & (df['Resolution'].isin(['640x640', '416x416']))]
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Inference Latency vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(res_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, f"{v}ms", ha='center')
    plt.savefig(os.path.join(plots_dir, 'latency_vs_resolution.png'))
plt.close()

# 2. FPS vs Resolution (for yolov8n, FP32)
plt.figure(figsize=(10, 6))
if not res_df.empty:
    plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
    plt.title('Inference FPS vs Input Resolution (YOLOv8n, FP32)')
    plt.xlabel('Resolution')
    plt.ylabel('FPS')
    for i, v in enumerate(res_df['Average_FPS']):
        plt.text(i, v + 0.2, f"{v} FPS", ha='center')
    plt.savefig(os.path.join(plots_dir, 'fps_vs_resolution.png'))
plt.close()

# 3. FPS vs Model (at 640x640, FP32)
plt.figure(figsize=(10, 6))
model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
if not model_df.empty:
    plt.bar(model_df['Model'], model_df['Average_FPS'], color=['skyblue', 'plum'])
    plt.title('Inference FPS: Nano vs Small (640x640, FP32)')
    plt.xlabel('Model Variant')
    plt.ylabel('FPS')
    for i, v in enumerate(model_df['Average_FPS']):
        plt.text(i, v + 0.1, f"{v} FPS", ha='center')
    plt.savefig(os.path.join(plots_dir, 'fps_vs_model.png'))
plt.close()

# 4. Latency: FP32 vs FP16 (YOLOv8n, 640x640)
plt.figure(figsize=(10, 6))
prec_df = df[(df['Model'] == 'yolov8n') & (df['Resolution'] == '640x640')]
if not prec_df.empty:
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['gold', 'lightgrey'])
    plt.title('Latency Comparison: FP32 vs FP16 (YOLOv8n, 640x640)')
    plt.xlabel('Precision')
    plt.ylabel('Latency (ms)')
    for i, v in enumerate(prec_df['Average_Latency_ms']):
        plt.text(i, v + 0.5, f"{v}ms", ha='center')
    plt.savefig(os.path.join(plots_dir, 'latency_precision_comp.png'))
plt.close()

print(f'Plots saved successfully in {plots_dir}')
