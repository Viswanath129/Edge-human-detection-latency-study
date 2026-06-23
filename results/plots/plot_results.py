import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the summary data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "tables", "summary.csv")

    if not os.path.exists(csv_path):
        print(f"Summary CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Set plot directory
    os.chdir(base_dir)

    # Plot 1: Average Latency vs Resolution (for yolov8n, FP32)
    plt.figure(figsize=(10, 6))
    res_df = df[(df['Model'] == 'yolov8n') & (df['Precision'] == 'FP32')]
    if not res_df.empty:
        plt.bar(res_df['Resolution'], res_df['Average_Latency_ms'], color=['salmon', 'lightblue'])
        plt.title('Average Latency vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(res_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_vs_resolution.png')
    plt.close()

    # Plot 2: Average FPS vs Resolution (for yolov8n, FP32)
    plt.figure(figsize=(10, 6))
    if not res_df.empty:
        plt.bar(res_df['Resolution'], res_df['Average_FPS'], color=['lightgreen', 'orange'])
        plt.title('Average FPS vs Input Resolution (YOLOv8n, FP32)')
        plt.xlabel('Input Resolution')
        plt.ylabel('Average FPS')
        for i, v in enumerate(res_df['Average_FPS']):
            plt.text(i, v + 0.2, str(v), ha='center')
        plt.savefig('fps_vs_resolution.png')
    plt.close()

    # Plot 3: FPS vs Model (at 640x640, FP32)
    plt.figure(figsize=(10, 6))
    model_df = df[(df['Resolution'] == '640x640') & (df['Precision'] == 'FP32')]
    if not model_df.empty:
        plt.bar(model_df['Model'], model_df['Average_FPS'], color=['purple', 'teal'])
        plt.title('Average FPS vs Model Size (640x640, FP32)')
        plt.xlabel('Model Variant')
        plt.ylabel('Average FPS')
        for i, v in enumerate(model_df['Average_FPS']):
            plt.text(i, v + 0.2, str(v), ha='center')
        plt.savefig('fps_vs_model.png')
    plt.close()

    # Plot 4: Latency/FPS comparison across precision levels (for yolov8n at 640x640)
    plt.figure(figsize=(10, 6))
    prec_df = df[(df['Resolution'] == '640x640') & (df['Model'] == 'yolov8n')]
    if len(prec_df) > 1:
        plt.bar(prec_df['Precision'], prec_df['Average_Latency_ms'], color=['grey', 'cyan'])
        plt.title('Inference Latency: FP32 vs FP16 (YOLOv8n, 640x640)')
        plt.xlabel('Precision Level')
        plt.ylabel('Average Latency (ms)')
        for i, v in enumerate(prec_df['Average_Latency_ms']):
            plt.text(i, v + 0.5, str(v), ha='center')
        plt.savefig('latency_precision_comp.png')
    plt.close()

    print('Plots updated successfully in results/plots')

if __name__ == "__main__":
    main()
