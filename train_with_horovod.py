import os
import json
import glob
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.utils import shuffle
from video_utils import load_data
import horovod.tensorflow.keras as hvd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

parent = os.path.dirname(os.getcwd())
videos = glob.glob(parent + "/train_sample_videos/*.mp4")[:100]
f = open(parent + "/train_sample_videos/metadata.json")
valid = json.load(f)
train_X, val_X, train_y, val_y = load_data(videos, valid, verbose=False)
train_X, train_y = shuffle(train_X, train_y, random_state=42)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

batch_size = 16
batch_size *= hvd.size()
epochs = 10
input_shape = (640, 360, 3)

optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001 * hvd.size(),
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

model = create_model()

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

model.fit(train_X, train_y, steps_per_epoch=len(train_X)//batch_size, epochs=epochs, validation_data=(val_X, val_y), callbacks=callbacks)
