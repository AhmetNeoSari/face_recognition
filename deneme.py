from face_recognition.arcface.utils import read_features
from dataclasses import field
import numpy as np


features_path = "./datasets/face_features/feature"


features = read_features(features_path)

images_name, images_emb = features
images_name = list(images_name)
images_name = np.array(images_name)

images_emb = list(images_emb)
images_emb = np.array(images_emb)

person_name = "ahmet_sarı"
indices_to_remove = [i for i, name in enumerate(images_name) if name == person_name]

for index in sorted(indices_to_remove, reverse=True):
    print("index", index)
    print(images_emb[index])

print(indices_to_remove)
