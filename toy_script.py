#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_observation.overview import summarize_data
from data_processing.preprocessing import preprocess_data
from data_processing.loader import load_data
from feature_engineering.extractor import extract_combined_features
from modeling.evaluation import train_evaluate_dt
from modeling.evaluation import train_evaluate_NeuralNet




if __name__ == '__main__':
    print("Loading data...")
    X_train, y_train, X_test = load_data(data_path='Data')
    #summarize_data(X_train)

    print("Preprocessing data...")
    print(f"Before preprocessing: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")
    X_train_cleaned, X_test_cleaned = preprocess_data(X_train, X_test)
    print(f"After preprocessing: X_train_cleaned shape = {X_train_cleaned.shape}, X_test_cleaned shape = {X_test_cleaned.shape}")


    # Feature Engineering
    print("Extracting features...")
    X_train_features = extract_combined_features(X_train_cleaned)
    X_test_features = extract_combined_features(X_test_cleaned)

    print(f"Training Features Shape: {X_train_features.shape}")
    print(f"Test Features Shape: {X_test_features.shape}")

    print("Evaluating Models on Traing Data")
    train_evaluate_dt(X_train_features, y_train, X_test_features)
    train_evaluate_NeuralNet(X_train_features, y_train, X_test_features)
    #write_submission(y_test)

    #clf = KNeighborsClassifier(n_neighbors=1)
    #clf.fit(X_train, y_train)

    #y_test = clf.predict(X_test)

    
