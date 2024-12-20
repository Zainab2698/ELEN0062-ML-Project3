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

    fft_magnitude = np.abs(np.fft.rfft(reshaped, axis=2))
    print(fft_magnitude.shape)
    mean_fft_magnitude = np.mean(fft_magnitude, axis=2) 
    std_fft_magnitude = np.std(fft_magnitude, axis=2)

    
    return np.hstack([mean, std, max_val, min_val,mean_fft_magnitude,std_fft_magnitude])