import cv2
import time
import pandas as pd
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # nano = fast (good for edge)

# Open webcam
cap = cv2.VideoCapture(0)

latencies = []
frame_count = 0
start_time = time.time()

INPUT_SIZE = 640  # change later to 416

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # Resize frame
    frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    # Run inference
    results = model(frame_resized, conf=0.4, iou=0.5, verbose=False)

    t1 = time.time()
    latency = (t1 - t0) * 1000  # ms
    latencies.append(latency)

    frame_count += 1

    # Display (optional)
    annotated = results[0].plot()
    cv2.imshow("Human Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()

# Metrics
avg_latency = sum(latencies) / len(latencies)
fps = frame_count / (end_time - start_time)

print(f"Avg Latency: {avg_latency:.2f} ms")
print(f"FPS: {fps:.2f}")

# Save results
df = pd.DataFrame({
    "latency_ms": latencies
})
df.to_csv("results/tables/resolution_{}_results.csv".format(INPUT_SIZE), index=False)
