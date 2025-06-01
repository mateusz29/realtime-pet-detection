import cv2
import streamlit as st
import torch
from ultralytics import YOLO

torch.classes.__path__ = []


@st.cache_resource
def load_model():
    return YOLO("model/yolo11s.pt")


model = load_model()

st.title("🐶🐱 Real-Time Pet Detector with YOLOv11")
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not available.")
        break

    results = model.predict(source=frame, conf=0.5, classes=None, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label not in ["cat", "dog"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

cap.release()
