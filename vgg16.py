import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "../dados"
MODELO_NOME = "vgg16"

# Carregar dados
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
).flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Modelo VGG16 — Transfer Learning
base_vgg = applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)
base_vgg.trainable = False

model = models.Sequential([
    base_vgg,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(f"resultados/{MODELO_NOME}_melhor.h5", save_best_only=True)
]

print(f"🚀 Treinando {MODELO_NOME.upper()}...")
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

import numpy as np
np.save(f"resultados/historico_{MODELO_NOME}.npy", history.history)
print(f"✅ {MODELO_NOME.upper()} concluído!")
