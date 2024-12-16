#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(data_path):
    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test

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

def train_evaluate(X_train, y_train):
    # Splitting training daat for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # The classifiers to be tested
    

def write_submission(y, submission_path='example_submission.csv'):
    parent_dir = os.path.dirname(submission_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(submission_path):
        os.remove(submission_path)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(submission_path, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print(f'Submission saved to {submission_path}.')

if __name__ == '__main__':
    X_train, y_train, X_test = load_data(data_path='./')

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    y_test = clf.predict(X_test)

    write_submission(y_test)
