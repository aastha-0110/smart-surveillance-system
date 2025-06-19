 Smart Surveillance System

A basic object detection system using Python, OpenCV, and YOLOv5. It detects objects from webcam, image, or video and logs the results in a CSV file.

Features
- Real-time object detection using webcam
- Detects objects like person, phone, etc.
- Logs detection time, object name, and confidence
- Saves results in a CSV file

 Requirements
- Python 3.x
- torch
- opencv-python
- pandas

 How to Run

1. Install dependencies:
pip install -r requirements.txt

2.Run the script:
python smart_object_detection.py

3.Press q to quit webcam window.

Output
Detections are saved to logs/detections.csv
