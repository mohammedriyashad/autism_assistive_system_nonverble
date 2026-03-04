import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# LOAD DATA
# ==========================
data = pd.read_csv("../dataset/hagrid/gesture_landmarks.csv")

X = data.drop("label", axis=1)
y = data["label"]

# ==========================
# ENCODE LABELS
# ==========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================
# TRAIN / TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ==========================
# EVALUATE
# ==========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "../dataset/gesture_model.pkl")
joblib.dump(le, "../dataset/label_encoder.pkl")

print("\nModel saved successfully!")