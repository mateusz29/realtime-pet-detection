# Real-Time Pet Detection with YOLOv11

A Streamlit app that uses YOLOv11 to detect pets (cats and dogs) in real time via your webcam. It supports live bounding boxes, snapshots, optional video recording, and FPS display.

## Features

* Real-time webcam detection (YOLOv11 via Ultralytics)
* Auto-recording when a pet appears
* Save snapshot with bounding boxes
* User-defined save folder
* FPS counter and confidence slider

## Requirements

* Python 3.8+
* YOLOv11 model file
* OpenCV
* Streamlit
* Ultralytics

## Usage

```bash
streamlit run app.py
```

Use the controls in the main panel to start webcam, set confidence, enable recording, or take snapshots.

## Model Weights

Download or train `yolo11s.pt` (YOLOv11 small) and place it under the `model/` directory:

```
model/yolo11s.pt
```

You can use [Ultralytics](https://github.com/ultralytics/ultralytics) to train or download the weights.

