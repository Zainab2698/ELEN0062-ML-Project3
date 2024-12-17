import os
import numpy as np

def load_data(data_path):
    FEATURES = range(2, 33) # create a sequence of int from 2 to 32 (number of sensors)
    N_TIME_SERIES = 3500

    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')

    X_train = np.zeros((N_TIME_SERIES, len(FEATURES) * 512)) #2D array
    X_test = np.zeros((N_TIME_SERIES, len(FEATURES) * 512))

    # Load data from the sensors to the traing DS X_train
    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f))) #selects a block of columns in X_train corresponding to feature f (if f=2, (2-2)*512:(2-2+1)*512->[:, 0:512])
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test