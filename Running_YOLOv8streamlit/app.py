import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Title and description
st.title("Object Detection WebApp with YOLOv8")
st.write("Upload images, videos, or use your webcam for real-time object detection.")

# Sidebar options
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video", "Webcam"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        image = np.array(image)
        results = model(image)

        # Show detection result
        st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            stframe.image(results[0].plot(), channels="BGR")
        cap.release()

elif option == "Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        stframe.image(results[0].plot(), channels="BGR")

    cap.release()
