# -*- coding: utf-8 -*-
from ultralytics import YOLO


nano_model = "yolov8n.pt"
small_model = "yolov8s.pt"
medium_model = "yolov8m.pt"
# Load the model.

model = YOLO("yolov8m.pt")

# Training.
results = model.train(
    data="custom_data.yaml",
    imgsz=640,
    epochs=10,
    batch=8,
    val=False,
    name="yolov8m_custom",
)
