import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Handle the missing values
def preprocess_data(X_train, X_test):
    # Replacing the missing values with NaN
    X_train[X_train == -999999.99] = np.nan
    X_test[X_train == -999999.99] = np.nan

    # Interpolating missing values
    X_train = np.apply_over_axes(lambda x: pd.Series(x).interpolate().to_numpy(), axis=0, arr=X_train)
    X_test = np.apply_over_axes(lambda x: pd.Series(x).interpolate().to_numpy(), axis=0, arr=X_test)

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def normalize_data(X_train, X_test):
    X_train[X_train == -999999.99] = 0
    X_test[X_test == -999999.99] = 0

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def normalize_data_with_mean(X_train, X_test):
    for X in [X_train, X_test]:
        for i in range(X.shape[1]):
            valid_values = X[:, i] != -999999.99
            feature_mean = np.mean(X[valid_values, i]) if np.any(valid_values) else 0
            X[~valid_values, i] = feature_mean

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
