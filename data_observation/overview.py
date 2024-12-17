import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_data(X, y=None, n_sensors=31, time_steps=512, missing_value=-999999.99):
    """
    Summarize the dataset with sensor-level stats
    """

    n_samples, n_features = X.shape
    print(f"Dataset Shape: {X.shape} (Samples: {n_samples}, Features: {n_features})")

    #Checking messing values
    missing_count = np.sum(X == missing_value)
    print(f"Missing Values: {missing_count} total")

    #Sensor-level summary
    print("\nSensor-Level Statistics:")
    sensor_missing_counts = []
    for sensor_idx in range(n_sensors):
        sensor_data = X[:, sensor_idx * time_steps:(sensor_idx + 1) * time_steps]
        missing_count = np.sum(sensor_data == missing_value)
        sensor_missing_counts.append(missing_count)

        # Visualize Missing Values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(1, n_sensors + 1)), y=sensor_missing_counts, palette="viridis")
    plt.title("Missing Values Per Sensor")
    plt.xlabel("Sensor Index")
    plt.ylabel("Number of Missing Values")
    plt.show()

    # --- Sensor Statistics ---
    sensor_means = []
    sensor_stds = []
    for sensor_idx in range(n_sensors):
        sensor_data = X[:, sensor_idx * time_steps:(sensor_idx + 1) * time_steps]
        sensor_data_cleaned = np.where(sensor_data == missing_value, np.nan, sensor_data)
        sensor_means.append(np.nanmean(sensor_data_cleaned))
        sensor_stds.append(np.nanstd(sensor_data_cleaned))

    # Visualize Sensor Statistics
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    sns.barplot(x=list(range(1, n_sensors + 1)), y=sensor_means, ax=ax[0], palette="Blues_d")
    ax[0].set_title("Mean Value Per Sensor")
    ax[0].set_xlabel("Sensor Index")
    ax[0].set_ylabel("Mean Value")   

    sns.barplot(x=list(range(1, n_sensors + 1)), y=sensor_stds, ax=ax[1], palette="Reds_d")
    ax[1].set_title("Standard Deviation Per Sensor")
    ax[1].set_xlabel("Sensor Index")
    ax[1].set_ylabel("Standard Deviation")
    plt.tight_layout()
    plt.show()


    # --- Sensor Distributions ---
    print("\nSensor Value Distributions (First 4 Sensors):")
    for sensor_idx in range(min(4, n_sensors)):  # Plot first 4 sensors
        sensor_data = X[:, sensor_idx * time_steps:(sensor_idx + 1) * time_steps]
        sensor_data_cleaned = np.where(sensor_data == missing_value, np.nan, sensor_data)
        plt.figure(figsize=(10, 4))
        sns.histplot(sensor_data_cleaned.flatten(), bins=50, kde=True)
        plt.title(f"Sensor {sensor_idx + 1} Value Distribution")
        plt.xlabel("Sensor Value")
        plt.ylabel("Frequency")
        plt.show()

    # --- Target Class Distribution ---
    if y is not None:
        print("\nTarget Class Distribution:")
        unique, counts = np.unique(y, return_counts=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=unique, y=counts, palette="coolwarm")
        plt.title("Target Class Distribution")
        plt.xlabel("Activity ID")
        plt.ylabel("Count")
        plt.show()

        