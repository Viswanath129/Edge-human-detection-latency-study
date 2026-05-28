import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Ensure plots directory exists
os.makedirs('.', exist_ok=True)

def plot_resolution_comparison(df):
    res_df = df[df['Observation'].str.contains('Baseline|Faster inference')]
    plt.figure(figsize=(10, 6))
    plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
    plt.title('Latency vs Input Resolution (YOLOv8n FP32)')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_vs_resolution.png')
    plt.close()

def plot_precision_comparison(df):
    # Filtering for FP32 vs FP16 at 640x640
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n') & (df['Observation'].str.contains('precision'))]
    if prec_df.empty: return

    plt.figure(figsize=(10, 6))
    plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['green', 'orange'])
    plt.yscale('log') # Use log scale because FP16 is much slower on CPU
    plt.title('Latency: FP32 vs FP16 (Log Scale - CPU)')
    plt.ylabel('Latency (ms)')
    plt.savefig('precision_comparison.png')
    plt.close()

def plot_model_size_comparison(df):
    size_df = df[df['Observation'].str.contains('model')]
    plt.figure(figsize=(10, 6))
    plt.bar(size_df['Model'], size_df['Average_Latency_ms'], color=['purple', 'cyan'])
    plt.title('Latency vs Model Size (640x640 FP32)')
    plt.ylabel('Latency (ms)')
    plt.savefig('model_size_comparison.png')
    plt.close()

plot_resolution_comparison(df)
plot_precision_comparison(df)
plot_model_size_comparison(df)

print('Updated plots saved in results/plots/')
