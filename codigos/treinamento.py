# codigos/treinamento.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)
BATCH = 32
EPOCHS = 20

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True).flow_from_directory(
    "../dados/train", target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "../dados/val", target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(3,activation='softmax')
])
model.compile(optimizer='adam', loss='categor
