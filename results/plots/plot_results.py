import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

def plot_metric(df_subset, category, metric, title, filename, color):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_subset['Variant'], df_subset[metric], color=color)
    plt.title(f'{title} for {category}')
    plt.xlabel('Variant')
    plt.ylabel(metric.replace('_', ' '))

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot for each category
categories = df['Category'].unique()
colors_fps = ['lightgreen', 'orange', 'cyan']
colors_latency = ['salmon', 'lightblue', 'plum']

for i, cat in enumerate(categories):
    df_cat = df[df['Category'] == cat]

    # FPS Plot
    plot_metric(df_cat, cat, 'Average_FPS', 'Average FPS', f'fps_{cat.lower()}.png', colors_fps[i % 3])

    # Latency Plot
    plot_metric(df_cat, cat, 'Average_Latency_ms', 'Average Latency (ms)', f'latency_{cat.lower()}.png', colors_latency[i % 3])

print('Plots saved successfully in results/plots')
