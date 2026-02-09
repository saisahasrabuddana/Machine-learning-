# KNN Classification Program

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 23],
    'Salary': [15000, 29000, 48000, 60000, 52000, 80000, 58000, 82000, 90000, 20000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Separate features and target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Step 4: Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 6: Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Step 7: Train the model
knn.fit(X_train, y_train)

# Step 8: Predict
y_pred = knn.predict(X_test)

# Step 9: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
