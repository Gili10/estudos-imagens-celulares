# codigos/treinamento.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "../dados"
MODELO = "cnn_propria"  # mude para "vgg16" ou "resnet50" para comparar

# Carregamento e aumento de dados
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
).flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = train_gen.num_classes

# Escolha da arquitetura
if MODELO == "cnn_propria":
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
elif MODELO == "vgg16":
    base = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
elif MODELO == "resnet50":
    base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

# Compilar e treinar
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(f"modelo_{MODELO}.h5", save_best_only=True)
]

print(f"🚀 Treinando: {MODELO}")
historico = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

print(f"✅ Concluído! Modelo salvo: modelo_{MODELO}.h5"
