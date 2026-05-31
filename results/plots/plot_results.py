import pandas as pd
import matplotlib.pyplot as plt
import os

# Use absolute path resolution for consistent result loading
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../tables/summary.csv')

# Load the summary data
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Create plots directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Generate labels for the X-axis: combining Resolution, Model, and Precision
df['Label'] = df['Resolution'] + '\n' + df['Model'] + '\n' + df['Precision']

# Plot 1: Average Latency vs Configuration
plt.figure(figsize=(12, 6))
plt.bar(df['Label'], df['Average_Latency_ms'], color='salmon')
plt.title('Inference Latency across Configurations')
plt.xlabel('Configuration (Resolution, Model, Precision)')
plt.ylabel('Average Latency (ms)')
plt.xticks(rotation=45)
for i, v in enumerate(df['Average_Latency_ms']):
    plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'latency_comparison.png'))
plt.close()

# Plot 2: Average FPS vs Configuration
plt.figure(figsize=(12, 6))
plt.bar(df['Label'], df['Average_FPS'], color='lightgreen')
plt.title('Inference FPS across Configurations')
plt.xlabel('Configuration (Resolution, Model, Precision)')
plt.ylabel('Average FPS')
plt.xticks(rotation=45)
for i, v in enumerate(df['Average_FPS']):
    plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'fps_comparison.png'))
plt.close()

print(f'Plots saved successfully in {base_dir}')
