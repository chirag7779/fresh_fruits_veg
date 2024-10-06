from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')
results=model('../Image/2.JPG', show=True)
cv2.waitKey(0)