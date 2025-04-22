import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
df = pd.read_csv("dataset/diabetes.csv")

# ------------------ EDA ------------------ #
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget column value counts:\n", df['Outcome'].value_counts())

# Create output folder if not exist
output_path = os.path.dirname(os.path.abspath(__file__))

# Plot target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df)
plt.title("Target Class Distribution (Outcome)")
plt.savefig(os.path.join(output_path, 'diabetes_target_distribution.png'))
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(output_path, 'diabetes_corr_heatmap.png'))
plt.close()

# ------------------ Model Training ------------------ #
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Find the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
best_accuracy = accuracies[best_model_name]
print(f"✅ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Save best model
model_path = os.path.join(output_path, "diabetes_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
