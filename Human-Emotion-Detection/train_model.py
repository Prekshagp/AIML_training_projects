import os
import joblib
import numpy as np
from utils.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = 'data/ravdess/'
labels, features = [], []

for file in os.listdir(DATASET_DIR):
    if file.endswith('.wav'):
        path = os.path.join(DATASET_DIR, file)
        label = file.split('-')[2]
        features.append(extract_features(path))
        labels.append(label)

X = np.array(features)
y = LabelEncoder().fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'model/emotion_model.pkl')
