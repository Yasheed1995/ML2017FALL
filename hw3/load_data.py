import numpy as np
from   keras.utils import np_utils
from   math import floor
import pandas as pd
import sys
import argparse
import random
import h5py

def load_data(f):

    y_train = np.array(f['label'].values)
    X_train = np.array(f['feature'].values)

    X_train, y_train, X_valid, y_valid \
            = split_valid_set(X_train, y_train, 0.2)

    X_train = (np.array([list(map(int, x.split(' '))) for x in X_train]))
    X_valid = (np.array([list(map(int, x.split(' '))) for x in X_valid]))

    y_train = np_utils.to_categorical(y_train)
    y_valid = np_utils.to_categorical(y_valid)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)

    #X_train = X_train.astype('float32')
    #y_train = y_train.astype('float32')

    X_train = np.array([x.reshape((48, 48)) for x in X_train]) \
                .reshape(-1, 48, 48, 1)
    X_valid = np.array([x.reshape((48, 48)) for x in X_valid]) \
                .reshape(-1, 48, 48, 1)

    #print (X_train)

    X_train = np.true_divide(X_train, 255)
    X_valid = np.true_divide(X_valid, 255)

    return (X_train, y_train), (X_valid, y_valid)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_valid, Y_valid = X_all[0: valid_data_size], Y_all[0: valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def main():
    f = pd.read_csv(sys.argv[1])
    (X_train, y_train), (X_valid, y_valid) = load_data(f)
    h5f = h5py.File('data/data.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_valid', data=X_valid)
    h5f.create_dataset('y_valid', data=y_valid)
    h5f.close()

if __name__ == '__main__':
    main()
