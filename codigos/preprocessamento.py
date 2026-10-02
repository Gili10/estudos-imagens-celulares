# codigos/preprocessamento.py
import os, shutil, random
from pathlib import Path

orig = Path("dados originais")
dest = Path("dados")
classes = ["sem_mascara", "com_mascara", "mascara_incorreta"]

for split in ["train", "val", "test"]:
    for c in classes:
        (dest/split/c).mkdir(parents=True, exist_ok=True)

for c in classes:
    imgs = list((orig/c).glob("."))
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n*0.7); n_val = int(n*0.15)
    splits = [("train", imgs[:n_train]), ("val", imgs[n_train:n_train+n_val]), ("test", imgs[n_train+n_val:])]
    for split, files in splits:
        for f in files:
            shutil.copy(f, dest/split/c/f.name)
    print(f"{c}: {n} imagens organizadas")
