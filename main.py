# Importing proper libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("abalone.csv")

# 1. Ordinary Least Squares; 5-fold cross validation

# Clean data (Drop missing, standardize, convert to numpy)
df.dropna(inplace = True)

X = df.drop(columns=['Rings', 'Sex'])
y = df['Rings']

standard = StandardScaler()
X_standard = standard.fit_transform(X)

X_standard = np.hstack([X_standard, np.ones((len(X_standard),1))])
y = y.to_numpy()

ols_error = []
for i, (train_index, test_index) in enumerate(kf.split(X_standard)):

    # Subsets X on the train and test
    X_train = X_standard[train_index]
    X_test = X_standard[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    # Linear Regression
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)

    # Generate Predictions for test
    yhat = np.dot(X_test, w)

    # Compute Squared Error:
    error = y_test - yhat
    error_sq = np.mean(error * error)
    ols_error.append(error_sq)

# Compute average RMSE over all folds
ols_error = np.sqrt(np.mean(ols_error))
print("OLS Error: ", ols_error)

# 2. One-Hot Encoding Error

# Clean data (Drop missing, standardize, convert to numpy)

# Add binary features for 'Sex'
df['Sex_M'] = (df['Sex'] == 'M').astype(int)
df['Sex_F'] = (df['Sex'] == 'F').astype(int)
df['Sex_I'] = (df['Sex'] == 'I').astype(int)

df.drop(columns=['Sex'], inplace = True)

X = df.drop(columns=['Rings'])
y = df['Rings']

standard = StandardScaler()
X_standard = standard.fit_transform(X)

X_standard = np.hstack([X_standard, np.ones((len(X_standard),1))])
y = y.to_numpy()

onehot_error = []
for i, (train_index, test_index) in enumerate(kf.split(X_standard)):

    # Subsets X on the train and test
    X_train = X_standard[train_index]
    X_test = X_standard[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    # Linear Regression
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)

    # Generate Predictions for test
    yhat = np.dot(X_test, w)

    # Compute Squared Error:
    error = y_test - yhat
    error_sq = np.mean(error * error)
    onehot_error.append(error_sq)

# Compute average RMSE over all folds
onehot_error = np.sqrt(np.mean(onehot_error))
print("Onehot Error: ", onehot_error)

# 3. Ridge Regression:
lambda_weight = 1e2

# Clean and preprocess data (reaed, missing, onehot)
df = pd.read_csv("../resource/asnlib/publicdata/abalone.csv")

df.dropna(inplace=True)

df['Sex_M'] = (df['Sex'] == 'M').astype(int)
df['Sex_F'] = (df['Sex'] == 'F').astype(int)
df['Sex_I'] = (df['Sex'] == 'I').astype(int)

df.drop(columns=['Sex'], inplace=True)

binary_features = df[['Sex_F', 'Sex_I', 'Sex_M']]
df.drop(columns=['Sex_F', 'Sex_I', 'Sex_M'], inplace=True)

standard = StandardScaler()
X_standard = standard.fit_transform(X)

X = df.drop(columns=['Rings'])
y = df['Rings']

X_standard = np.hstack([X_standard, binary_features, np.ones((len(X_standard), 1))])
y = y.to_numpy()

ridge_error_store = []
for i, (train_index, test_index) in enumerate(kf.split(X_standard)):

    # Subsets X on the train and test
    X_train = X_standard[train_index]
    X_test = X_standard[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    # Ridge regression
    lambda_matrix = lambda_weight * np.identity(X_train.shape[1])
    ridge_coef_matrix = np.dot(X_train.T, X_train) + lambda_matrix

    # Train the model using ridge regression
    w = np.dot(np.dot(np.linalg.inv(ridge_coef_matrix), X_train.T), y_train)

    # Generate predictions for the test set
    yhat = np.dot(X_test, w)

    # Compute the squared error and add
    error = y_test - yhat
    error_sq = np.mean(error * error)
    ridge_error_store.append(error_sq)

# Compute the average RMSE over all folds for ridge regression
ridge_error = np.sqrt(np.mean(ridge_error_store))
print(ridge_error)
