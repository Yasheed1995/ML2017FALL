import pandas as pd
import numpy as np
from util import DataManager
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import sys
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from keras.models import load_model

sys.path.append('./data')

n_batch=2048

def main(_model):

    dm = DataManager()
    dm.read_data(sys.argv[1], 'test', with_label=False)
    model = load_model('model/model.hdf5')
    test_y = model.predict({'user_input':dm.data['test'][0][:,0],
                            'movie_input':dm.data['test'][0][:,1]},
                          batch_size=n_batch, verbose=1)
    # data normalization
    #test_y = test_y * dm.train_rating_std + dm.train_rating_mean
    #np.save('test_y_1_1024.npy', test_y)
    dm.write_file(test_y, sys.argv[2])

if __name__ == '__main__':
    main(1)
