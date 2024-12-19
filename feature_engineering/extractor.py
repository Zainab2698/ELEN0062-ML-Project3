import numpy as np

N_SENSORS = 31
TIME_STEPS = 512
def extract_basic_features(X):
    n_samples, n_features = X.shape
    
    reshaped = X.reshape(n_samples, N_SENSORS, TIME_STEPS)
    mean = np.mean(reshaped, axis=2)
    std = np.std(reshaped, axis=2)
    max_val = np.max(reshaped, axis=2)
    min_val = np.min(reshaped, axis=2)
    return np.hstack([mean, std, max_val, min_val])

# Extract frequency features using FFT for each sensor's time series
def extract_frequency_features(X):
    n_samples, n_features = X.shape
    
    # Reshape data to (n_samples, n_sensors, time_steps)
    reshaped = X.reshape(n_samples, N_SENSORS, TIME_STEPS)

    # Compute FFT and take the magnitude of the first half of frequencies
    fft_features = np.abs(np.fft.fft(reshaped, axis=2))[:, :, :256]

    # Compute mean and standard deviation of FFT magnitudes we extracted
    mean_fft = np.mean(fft_features, axis=2)
    std_fft = np.std(fft_features, axis=2)

    # Concatenate frequency features
    features = np.hstack([mean_fft, std_fft])

    return features

def extract_combined_features(X):
    statistical_features = extract_basic_features(X)

    # Extract frequency features
    frequency_features = extract_frequency_features(X)

    # Combine all features
    combined_features = np.hstack([statistical_features, frequency_features])

    return combined_features