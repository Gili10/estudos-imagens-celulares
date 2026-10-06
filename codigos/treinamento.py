import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Configurações
IMG_SIZE = (128, 128)
BASE_DIR = "../dados"
NUM_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 32

# Carregar dados
print("📂 Carregando dados...")
X_train = np.load(f"{BASE_DIR}/X_train.npy")
X_val = np.load(f"{BASE_DIR}/X_val.npy")
y_train = np.load(f"{BASE_DIR}/y_train.npy")
y_val = np.load(f"{BASE_DIR}/y_val.npy")

print(f"Treino: {X_train.shape} - {y_train.shape}")
print(f"Validação: {X_val.shape} - {y_val.shape}")

# Criar modelo CNN própria
print("\n🧠 Criando modelo CNN própria...")

modelo = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# Criar pasta de resultados
os.makedirs("resultados", exist_ok=True)

# Treinar
print("\n🚀 Iniciando treinamento...")
historico = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Salvar modelo
modelo.save("resultados/cnn_propria_melhor.h5")
print("\n✅ Modelo salvo em resultados/cnn_propria_melhor.h5")
print(f"✅ Treinamento concluído!")
