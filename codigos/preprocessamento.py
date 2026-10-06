import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# Configurações
IMG_SIZE = (128, 128)
BASE_DIR = "../dados"
classes = ["celula_viva", "celula_morta", "contaminada"]

# Criar pastas se não existirem
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Carregar e processar imagens
dados, rotulos = [], []

for idx, cls in enumerate(classes):
    caminhos = glob(f"{BASE_DIR}/imagens_originais/{cls}/*")
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dados.append(img / 255.0)
        rotulos.append(idx)

# Dividir treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    np.array(dados), np.array(rotulos),
    test_size=0.2, random_state=42, stratify=rotulos
)

# Salvar arrays
np.save(f"{BASE_DIR}/X_train.npy", X_train)
np.save(f"{BASE_DIR}/X_val.npy", X_val)
np.save(f"{BASE_DIR}/y_train.npy", y_train)
np.save(f"{BASE_DIR}/y_val.npy", y_val)

print(f"✅ Pré-processamento concluído!")
print(f"   Treino: {len(X_train)} imagens")
print(f"   Validação: {len(X_val)} imagens")
