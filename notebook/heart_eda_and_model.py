import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('heart.csv')

# --------------------- EDA ---------------------

print("\n✅ First 5 Rows of the Dataset:")
print(df.head())

print("\n✅ Dataset Info:")
print(df.info())

print("\n✅ Null Values:")
print(df.isnull().sum())

print("\n✅ Descriptive Stats:")
print(df.describe())

# Target value distribution
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.savefig("notebook/heart_target_distribution.png")
plt.clf()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("notebook/heart_corr_heatmap.png")
plt.clf()

# --------------------- Preprocessing ---------------------

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------- Model Training ---------------------

models = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print(f"\n🔥 Best Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")

# --------------------- Save Best Model ---------------------

model_path = 'notebook/heart_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Model saved to {model_path}")
