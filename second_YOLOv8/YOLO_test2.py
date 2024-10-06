import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model (use your trained model path here)
model = YOLO('../fresh_fruits_veg/second_YOLOv8/best.pt')


# Set up the Streamlit interface
st.title("Object Detection WebApp with YOLOv8")
st.subheader("Upload an image, video or use your webcam to detect objects")

# Sidebar for options
option = st.sidebar.selectbox('Choose Input Source', ('Image', 'Video', 'Webcam'))


# Function to process and display image results
def process_image(image):
    # Convert PIL image to numpy array
    img_np = np.array(image)

    # Run YOLOv8 inference on the image
    results = model(img_np)

    # Display results on Streamlit
    st.image(results[0].plot(), caption='Detected Image', use_column_width=True)


# Function to process and display video results
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)

        # Display the frame with detections
        stframe.image(results[0].plot(), channels="BGR", use_column_width=True)

    cap.release()


# Function to process webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)  # Use webcam 0
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)

        # Display the frame with detections
        stframe.image(results[0].plot(), channels="BGR", use_column_width=True)

    cap.release()


# Handle different input sources
if option == 'Image':
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        process_image(image)

elif option == 'Video':
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        process_video(uploaded_video)

elif option == 'Webcam':
    st.write("Using webcam for object detection")
    process_webcam()
