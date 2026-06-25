"""
Artificial Neural Network (ANN)
Customer Churn Prediction

Compatible with:
- Python 3.10+
- TensorFlow 2.x / Keras 3.x
"""

# ============================================================
# Import Libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ============================================================
# Load Dataset
# ============================================================

dataset = pd.read_csv(
    r"D:\DS_PRACTICE\24-06-2026\ann\ANN_2nd\Churn_Modelling.csv"
)

# Independent Variables
X = dataset.iloc[:, 3:13]

# Dependent Variable
y = dataset.iloc[:, 13]

# ============================================================
# One Hot Encoding
# ============================================================

geography = pd.get_dummies(
    X["Geography"],
    drop_first=True,
    dtype=int
)

gender = pd.get_dummies(
    X["Gender"],
    drop_first=True,
    dtype=int
)

X = pd.concat(
    [X, geography, gender],
    axis=1
)

X.drop(
    ["Geography", "Gender"],
    axis=1,
    inplace=True
)

print("\nDataset Shape:", X.shape)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# Build ANN Model
# ============================================================

model = Sequential()

# Input Layer
model.add(Input(shape=(X_train.shape[1],)))

# Hidden Layer 1
model.add(Dense(
    units=11,
    activation="relu",
    kernel_initializer="he_uniform"
))

# Optional Dropout
model.add(Dropout(0.20))

# Hidden Layer 2
model.add(Dense(
    units=11,
    activation="relu",
    kernel_initializer="he_normal"
))

model.add(Dropout(0.20))

# Hidden Layer 3
model.add(Dense(
    units=15,
    activation="relu",
    kernel_initializer="he_normal"
))

# Output Layer
model.add(Dense(
    units=1,
    activation="sigmoid",
    kernel_initializer="glorot_uniform"
))

# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer="Adamax",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# Model Summary
# ============================================================

model.summary()

# ============================================================
# Train Model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.33,
    epochs=10,
    batch_size=10,
    verbose=1
)

# ============================================================
# Save Model
# ============================================================

model.save("customer_churn_ann.keras")

print("\nModel Saved Successfully")

# ============================================================
# Plot Accuracy
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# Plot Loss
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# Prediction
# ============================================================

y_prob = model.predict(X_test, verbose=0)

y_pred = (y_prob > 0.5).astype(int)

# ============================================================
# Evaluation
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")

print(cm)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")

print(f"{accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ============================================================
# Predict Single Customer
# ============================================================

sample_customer = np.array([[
    600,
    1,
    40,
    3,
    60000,
    2,
    1,
    1,
    50000,
    1,
    0
]])

sample_customer = scaler.transform(sample_customer)

prediction = model.predict(sample_customer, verbose=0)

print("\nSingle Customer Prediction")

if prediction[0][0] > 0.5:
    print("Customer is likely to Leave.")
else:
    print("Customer is likely to Stay.")

print(f"Probability: {prediction[0][0]:.4f}")