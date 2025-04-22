import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Create folders if not exist
"""os.makedirs("../notebook/models", exist_ok=True)
os.makedirs("../notebook/plots", exist_ok=True)"""

# Load dataset
df = pd.read_csv('dataset/parkinsons.csv')
print("First 5 rows:\n", df.head(), "\n")

# Drop irrelevant column
df.drop('name', axis=1, inplace=True)

# Features and target
X = df.drop('status', axis=1)
y = df['status']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Classifiers
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

accuracies = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
    except Exception as e:
        print(f"{name} training failed: {e}")

# Save best model
with open("parkinson_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model saved with accuracy: {best_accuracy:.4f}")

# Plot comparison
plt.figure(figsize=(8, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("parkinsons_model_accuracy.png")
plt.close()
