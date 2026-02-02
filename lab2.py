import pandas as pd
import numpy as np

# Create dataset
data = {
    'Age': [25, np.nan, 45, 120],
    'Salary': [40000, 55000, np.nan, 50000],
    'Purchased': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# a. Attribute Selection
df = df[['Age', 'Salary']]

# b. Handling Missing Values (using mean)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# c. Elimination of Outliers
df = df[df['Age'] < 100]

# d. Discretization
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 30, 60, 100],
    labels=['Young', 'Adult', 'Senior']
)

print("\nFinal Preprocessed Dataset:")
print(df)