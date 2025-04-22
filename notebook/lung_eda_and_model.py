# lung_eda_and_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('dataset/lung.csv')

# Show first 5 rows
print("First 5 rows:\n", df.head(), "\n")

# Dataset info
print("Dataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Target variable: 'Level'
print("\nTarget column value counts:\n", df['Level'].value_counts())

# Drop unnecessary columns
df.drop(['index', 'Patient Id'], axis=1, inplace=True)

# Encode target labels
le = LabelEncoder()
df['Level'] = le.fit_transform(df['Level'])  # Low = 1, Medium = 2, High = 0 (depending on fit)

# Split data
X = df.drop('Level', axis=1)
y = df['Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Print best model
print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Save best model
model_path = 'notebook/lung_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)
