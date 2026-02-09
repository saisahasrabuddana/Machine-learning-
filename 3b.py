# KNN Regression Program

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create regression dataset
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Create KNN regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Step 4: Train the model
knn_reg.fit(X_train, y_train)

# Step 5: Predict output values
y_pred = knn_reg.predict(X_test)

# Step 6: Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
