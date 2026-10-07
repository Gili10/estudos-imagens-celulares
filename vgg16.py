import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "../dados"
MODELO_NOME = "vgg16"

os.makedirs("resultados", exist_ok=True)

print("📂 Carregando dados...")
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

print(f"✅ Classes: {list(train_gen.class_indices.keys())}")

print(f"\n🧠 Construindo VGG16...")
base_vgg = applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_vgg.trainable = False

model = models.Sequential([
    base_vgg,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print(f"\n🚀 Treinando - {MODELO_NOME.upper()}")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(f"resultados/{MODELO_NOME}_melhor.h5", save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

np.save(f"resultados/historico_{MODELO_NOME}.npy", history.history)
print(f"\n✅ FINALIZADO! Val acc: {max(history.history['val_accuracy']):.4f}")
