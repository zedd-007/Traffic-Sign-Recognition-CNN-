# model_training/train_model.py

import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path to training images
DATASET_DIR = "C:/Users/Zaid Chikte/Desktop/Traffic Sign Recognition (CNN)/dataset/Train"
IMG_HEIGHT, IMG_WIDTH = 32, 32

# Load images and labels
images = []
labels = []

print("[INFO] Loading images...")
for label in tqdm(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(int(label))
            except Exception as e:
                print(f"[WARNING] Skipped: {img_file} ({e})")

# Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Normalize pixel values
X = X / 255.0

# One-hot encode labels
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("[INFO] Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Save model
MODEL_PATH = "../model/traffic_sign_cnn.h5"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")

# Save label map
label_map = {i: str(i) for i in range(num_classes)}
LABEL_MAP_PATH = "../model/label_map.json"
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f)
print(f"[INFO] Label map saved to {LABEL_MAP_PATH}")