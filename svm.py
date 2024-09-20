import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

# Define paths
train_dir = 'D:\\internship\\train'  # Adjust this to your actual path


# Initialize lists for images and labels
images = []
labels = []

# Load images and labels
for filename in tqdm(os.listdir(train_dir)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize to 64x64
        images.append(img.flatten())  # Flatten the image
        label = 1 if 'dog' in filename else 0  # Label: 1 for dog, 0 for cat
        labels.append(label)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Check counts of labels
unique, counts = np.unique(labels, return_counts=True)
print("Label counts:", dict(zip(unique, counts)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check classes in training and test sets
print(f"Classes in training set: {np.unique(y_train)}")
print(f"Classes in test set: {np.unique(y_test)}")

# Create a pipeline with scaling and SVM
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
