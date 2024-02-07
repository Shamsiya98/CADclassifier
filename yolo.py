import os
from ultralytics import YOLO


model=YOLO('yolov8n-cls.pt')

img_size=112
model.train(data=r"C:\Users\HP\Desktop\CAD classification\FDS",epochs=10,imgsz=img_size)
