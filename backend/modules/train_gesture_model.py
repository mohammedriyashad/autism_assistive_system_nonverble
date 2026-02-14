import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

dataset_path = "../dataset/gesture"

X = []
y = []

label_map = {}
label_id = 0

for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        gesture_name = file.replace(".csv", "")
        label_map[label_id] = gesture_name

        data = pd.read_csv(os.path.join(dataset_path, file), header=None)

        X.extend(data.values)
        y.extend([label_id] * len(data))

        label_id += 1

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/gesture_model.pkl")
joblib.dump(label_map, "../models/label_map.pkl")

print("Model Trained Successfully")