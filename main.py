import datetime
import os
import time

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
run = st.checkbox("🎥 Start Webcam")
enable_recording = st.checkbox("📼 Enable Recording")
snapshot_btn = st.button("📸 Save Snapshot")
confidence = st.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)
output_folder = st.text_input("Save Folder Path", "recordings")

FRAME_WINDOW = st.image([])
fps_display = st.empty()

cap = cv2.VideoCapture(0)
prev_time = time.time()

recording = False
recording_writer = None
recording_start_time = None
last_seen_time = None
snapshot_taken = False

os.makedirs(output_folder, exist_ok=True)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not available.")
        break

    raw_frame = frame.copy()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model.predict(source=frame, conf=confidence, classes=None, verbose=False)[0]

    pet_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ["cat", "dog"]:
            pet_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if snapshot_btn and not snapshot_taken:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"snapshot_{timestamp}.jpg"
        path = os.path.join(output_folder, filename)
        cv2.imwrite(path, frame)
        st.success(f"📸 Snapshot saved: {filename}")
        snapshot_taken = True
    elif not snapshot_btn:
        snapshot_taken = False

    if enable_recording:
        if pet_detected:
            last_seen_time = current_time
            if not recording:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"pet_{timestamp}.mp4"
                filepath = os.path.join(output_folder, filename)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                recording_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
                recording = True
                st.info(f"🎬 Started recording: {filename}")

        if recording and (last_seen_time is not None) and (current_time - last_seen_time > 2):
            recording = False
            recording_writer.release()
            recording_writer = None
            st.info("🛑 Stopped recording")

        if recording and recording_writer:
            recording_writer.write(raw_frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)
    fps_display.text(f"📈 FPS: {fps:.2f}")

cap.release()
