import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
IMG_SIZE = (128, 128)
BASE_DIR = "../dados"
classes = ["celula_viva", "celula_morta", "contaminada"]
modelos = ["cnn_propria", "vgg16"]

# Carregar dados
X_val = np.load(f"{BASE_DIR}/X_val.npy")
y_val = np.load(f"{BASE_DIR}/y_val.npy")

# Avaliar cada modelo
resultados = {}

for nome in modelos:
    caminho_modelo = f"resultados/{nome}_melhor.h5"
    if not os.path.exists(caminho_modelo):
        print(f"⚠️ Modelo {nome} não encontrado — pulando...")
        continue

    modelo = tf.keras.models.load_model(caminho_modelo)
    perdida, acuracia = modelo.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(modelo.predict(X_val, verbose=0), axis=1)

    resultados[nome] = {
        "acuracia": acuracia,
        "perdida": perdida,
        "y_pred": y_pred
    }

    print(f"\n📊 {nome.upper()}")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Perda: {perdida:.4f}")
    print(classification_report(y_val, y_pred, target_names=classes, digits=4))

    # Matriz de confusão
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusão — {nome}")
    plt.ylabel("Real")
    plt.xlabel("Previsto")
    plt.tight_layout()
    plt.savefig(f"resultados/matriz_{nome}.png", dpi=150)
    plt.close()

# Comparativo final
if resultados:
    print("\n" + "="*50)
    print("📋 COMPARATIVO DE MODELOS")
    print("="*50)
    for nome, res in resultados.items():
        print(f"{nome:15} | Acurácia: {res['acuracia']:.4f}")
        
