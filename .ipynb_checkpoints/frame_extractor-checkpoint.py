# Import libraries
import os
import cv2
import numpy as np

def run_extraction(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Loop through all images and strip out faces
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        count += 1
        yield frame
