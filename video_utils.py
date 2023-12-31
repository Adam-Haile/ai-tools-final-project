"""
video_utils.py - Adam Haile
Video Preprocessing Utilities for Deepfake CNN Project
"""
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(directories, valid, num_frames=30, test_size=0.2, val_size=0.1, verbose=True):
    train_X, train_y, test_X, test_y, val_X, val_y = [], [], [], [], [], []
    test_amount = int(len(directories) * test_size)
    val_amount = int(len(directories) * val_size)
    
    test_videos = random.sample(directories, test_amount)
    train_videos = [item for item in directories if item not in test_videos]
    
    val_videos = random.sample(train_videos, val_amount)
    train_videos = [item for item in train_videos if item not in val_videos]
    original_train_size = len(train_videos)
    original_test_size = len(test_videos)
    original_val_size = len(val_videos)
    
    if verbose:
        print("Processing train videos...")
        fake_videos = []
        real_videos = []
        for video in train_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        train_videos = real_videos
        random.shuffle(train_videos)
        print(f"Oversampled training videos. Original size: {original_train_size}, New size: {len(train_videos)}")
        for directory in tqdm(train_videos):
            frames = _load_and_preprocess_video(directory, num_frames)
            train_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                train_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
                
        print("Processing test videos...")
        fake_videos = []
        real_videos = []
        for video in test_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        test_videos = real_videos
        random.shuffle(test_videos)
        print(f"Oversampled testing videos. Original size: {original_test_size}, New size: {len(test_videos)}")
        for directory in tqdm(test_videos):
            frames = _load_and_preprocess_video(directory, num_frames)
            test_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                test_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
                
        print("Processing validation videos...")
        fake_videos = []
        real_videos = []
        for video in val_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        val_videos = real_videos
        random.shuffle(val_videos)
        print(f"Oversampled validation videos. Original size: {original_val_size}, New size: {len(val_videos)}")
        for directory in tqdm(val_videos):
            frames = _load_and_preprocess_video(directory, num_frames)
            val_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                val_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
    else:
        fake_videos = []
        real_videos = []
        for video in train_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        train_videos = real_videos
        random.shuffle(train_videos)
        print(f"Oversampled training videos. Original size: {original_train_size}, New size: {len(train_videos)}")
        for directory in train_videos:
            frames = _load_and_preprocess_video(directory, num_frames)
            train_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                train_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
                
        fake_videos = []
        real_videos = []
        for video in test_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        test_videos = real_videos
        random.shuffle(test_videos)
        print(f"Oversampled testing videos. Original size: {original_test_size}, New size: {len(test_videos)}")
        for directory in test_videos:
            frames = _load_and_preprocess_video(directory, num_frames)
            test_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                test_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))
                
        fake_videos = []
        real_videos = []
        for video in val_videos:
            video_name = video.split("/")[-1]
            if valid[video_name]["label"] == "FAKE":
                fake_videos.append(video)
            else:
                real_videos.append(video)

        oversample_factor = len(fake_videos) // len(real_videos)
        oversampled_list = random.choices(real_videos, k=len(real_videos) * oversample_factor)
        real_videos = oversampled_list[:len(fake_videos)]
        real_videos.extend(fake_videos)
        val_videos = real_videos
        random.shuffle(val_videos)
        print(f"Oversampled validation videos. Original size: {original_val_size}, New size: {len(val_videos)}")
        for directory in val_videos:
            frames = _load_and_preprocess_video(directory, num_frames)
            val_X.extend(frames)
            video_name = directory.split("/")[-1]
            for _ in frames:
                val_y.append(np.array([1, 0]) if valid[video_name]["label"] == "FAKE" else np.array([0, 1]))

    return np.array(train_X), np.array(test_X), np.array(val_X), np.array(train_y), np.array(test_y), np.array(val_y)

def _load_and_preprocess_video(video_path, num_frames, img_height=360, img_width=640):
    frames = []
    cap = cv2.VideoCapture(video_path)

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = _rotate_vertical(frame)
        frame = cv2.resize(frame, (img_width, img_height))
        frame = img_to_array(frame)
        frame = frame.astype('float16') / 255.0
        frames.append(frame)

    cap.release()
    return frames


def _rotate_vertical(x):
    if x.shape == (1920,1080,3):
        return np.rot90(x)
    else:
        return x
