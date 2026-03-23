# Step 1: Import required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 2: Create sample dataset
# Features (X)
X = np.array([[1, 20],
              [2, 21],
              [3, 22],
              [4, 23],
              [5, 24],
              [6, 25]])

# Labels (y)
y = np.array([0, 0, 0, 1, 1, 1])

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Step 8: Print results
print("Predicted Output:", y_pred)
print("Actual Output:", y_test)
print("Accuracy:", accuracy)