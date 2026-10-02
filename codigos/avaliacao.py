# codigos/avaliacao.py
import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (128,128)
model = load_model("../resultados/modelo_cnn_propria.h5")
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "../dados/test", target_size=IMG_SIZE, batch_size=32, class_mode='categorical', shuffle=False)

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.savefig("../resultados/matriz_confusao.png")
print("Avaliação salva em resultados/
