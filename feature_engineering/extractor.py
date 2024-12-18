import numpy as np

def extract_basic_features(X):
    n_samples, n_features = X.shape
    reshaped = X.reshape(n_samples, -1, 512)
    mean = np.mean(reshaped, axis=2)
    std = np.std(reshaped, axis=2)
    max_val = np.max(reshaped, axis=2)
    min_val = np.min(reshaped, axis=2)
    return np.hstack([mean, std, max_val, min_val])

def extract_basic_features1(X):
    n_samples, n_features = X.shape
    reshaped = X.reshape(n_samples, -1, 512)
    mean = np.mean(reshaped, axis=2)
    std = np.std(reshaped, axis=2)
    max_val = np.max(reshaped, axis=2)
    min_val = np.min(reshaped, axis=2)

    max_freq_val = []
    for i in range(n_samples):
        sample_max_freq = []
        for j in range(reshaped.shape[1]):
            values, counts = np.unique(reshaped[i, j, :], return_counts=True)
            sample_max_freq.append(values[np.argmax(counts)])
        max_freq_val.append(sample_max_freq)
    max_freq_val = np.array(max_freq_val)
    
    return np.hstack([mean, std, max_val, min_val, max_freq_val])