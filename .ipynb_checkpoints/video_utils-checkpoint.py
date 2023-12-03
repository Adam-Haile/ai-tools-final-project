"""
video_utils.py - Adam Haile
Video Preprocessing Utilities for Deepfake CNN Project
"""
import os
import cv2
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(directories, valid, num_frames=30, test_size=0.2, verbose=True):
    X, y = [], []

    if verbose:
        for directory in tqdm(directories):
            frames = _load_and_preprocess_video(directory, num_frames)
            X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
    else:
        for directory in directories:
            frames = _load_and_preprocess_video(directory, num_frames)
            X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
        
    print("Creating train/test split")
    train_X, val_X, train_y, val_y = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=42)

    return train_X, val_X, train_y, val_y

def _load_and_preprocess_video(video_path, num_frames, img_height=360, img_width=640):
    frames = []
    cap = cv2.VideoCapture(video_path)

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = _rotate_vertical(frame)
        frame = cv2.resize(frame, (img_height, img_width))
        frame = img_to_array(frame)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()
    return frames


def _rotate_vertical(x):
    if x.shape == (1920,1080,3):
        return np.rot90(x)
    else:
        return x
