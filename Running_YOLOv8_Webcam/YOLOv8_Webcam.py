from ultralytics import YOLO
import cv2
import os

# Use an absolute path to the yolov8.pt weights file
weights_path = os.path.join("C:\\", "Users", "chira", "PycharmProjects", "pythonProject2", "YOLO-Weights", "yolov8n.pt")

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Ensure the file exists before initializing the model
if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
else:
    model = YOLO(weights_path)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    out.release()
    cv2.destroyAllWindows()
