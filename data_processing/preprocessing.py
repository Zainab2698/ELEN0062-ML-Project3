import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):
    """
    Preprocesses training and test data by handling missing values and normalizing features.

    """
    # Ensure data is 2D and has correct shape
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)
    
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original X_test shape: {X_test.shape}")

    # Check for correct feature count
    if X_train.shape[1] != 15872 or X_test.shape[1] != 15872:
        raise ValueError("The input data does not have the expected 15,872 features. Check data loading!")

    # Replace missing values (-999999.99) with NaN
    X_train[X_train == -999999.99] = np.nan
    X_test[X_test == -999999.99] = np.nan

    # Interpolate missing values along each row
    X_train = pd.DataFrame(X_train).interpolate(axis=1, limit_direction='both').to_numpy()
    X_test = pd.DataFrame(X_test).interpolate(axis=1, limit_direction='both').to_numpy()

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Processed X_train shape: {X_train.shape}")
    print(f"Processed X_test shape: {X_test.shape}")

    return X_train, X_test
