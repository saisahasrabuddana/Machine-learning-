# Step 1: Import required libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Step 2: Create dataset
# X represents the independent variable (feature)
X = np.array([[1], [2], [3], [4], [5]])

# y represents the dependent variable (target value)
y = np.array([5, 7, 9, 11, 13])

# Step 3: Create Decision Tree Regressor model
model = DecisionTreeRegressor()

# Step 4: Train the model
model.fit(X, y)

# Step 5: Predict a value
prediction = model.predict([[6]])

# Step 6: Display result
print("Predicted value:", prediction)