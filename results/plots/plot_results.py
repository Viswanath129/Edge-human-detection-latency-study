import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the summary data
df = pd.read_csv('../tables/summary.csv')

# Create plots directory if it doesn't exist
os.makedirs('.', exist_ok=True)

# Plot 1: Average Latency vs Resolution
plt.figure(figsize=(8, 5))
plt.bar(df['Resolution'], df['Average_Latency_ms'], color=['salmon', 'lightblue'])
plt.title('Average Latency vs Input Resolution')
plt.xlabel('Input Resolution')
plt.ylabel('Average Latency (ms)')
for i, v in enumerate(df['Average_Latency_ms']):
    plt.text(i, v + 2, str(v), ha='center')
plt.savefig('latency_vs_resolution.png')
plt.close()

# Plot 2: Average FPS vs Resolution
plt.figure(figsize=(8, 5))
plt.bar(df['Resolution'], df['Average_FPS'], color=['lightgreen', 'orange'])
plt.title('Average FPS vs Input Resolution')
plt.xlabel('Input Resolution')
plt.ylabel('Average FPS (Frames/Sec)')
for i, v in enumerate(df['Average_FPS']):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.savefig('fps_vs_resolution.png')
plt.close()

print('Plots saved successfully in results/plots')
