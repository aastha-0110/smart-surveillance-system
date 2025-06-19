import cv2
import torch
import pandas as pd
from datetime import datetime
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
log_file = f"logs/detections_{today}.csv"


# Initialize CSV log file
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Timestamp", "Object", "Confidence"])
    df.to_csv(log_file, index=False)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()
print("✅ Surveillance started. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get pandas DataFrame

    for _, row in detections.iterrows():
        obj_name = row['name']
        confidence = row['confidence']

        # Log only high-confidence detections
        if confidence >= 0.5:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = pd.DataFrame([[now, obj_name, round(confidence, 2)]], columns=["Timestamp", "Object", "Confidence"])
            log_entry.to_csv(log_file, mode='a', header=False, index=False)

    # Annotate and show frame
    annotated_frame = results.render()[0]
    cv2.imshow("Smart Surveillance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting surveillance.")
        break

cap.release()
cv2.destroyAllWindows()