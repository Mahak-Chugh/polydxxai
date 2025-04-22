import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('dataset/kidney_disease.csv')

print("First 5 rows:")
print(df.head(), '\n')

print("Dataset Info:")
print(df.info(), '\n')

# Check for missing values
print("Missing values:")
print(df.isnull().sum(), '\n')

# Drop ID column
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Fix incorrect string in target column
df['classification'] = df['classification'].replace({'ckd\t': 'ckd'})

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Final cleanup: remove any residual NaN or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

best_acc = 0
best_model = None

"""for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model"""

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
    except Exception as e:
        print(f"Error with {name}: {e}")


# Save best model
with open('notebook/kidney_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nBest model saved as kidney_model.pkl with accuracy:", round(best_acc, 4))
