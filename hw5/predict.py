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
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
'''
sys.path.append('./data')

n_batch=2048

def main(_model):

    dm = DataManager()
    dm.read_data('data/test.csv', 'test', with_label=False)
    model = load_model('model/model_1_128.hdf5')
    test_y = model.predict({'user_input':dm.data['test'][0][:,0],
                            'movie_input':dm.data['test'][0][:,1]},
                          batch_size=n_batch, verbose=1)
    # data normalization
    #test_y = test_y * dm.train_rating_std + dm.train_rating_mean
    np.save('test_y_1_1024.npy', test_y)
    dm.write_file(test_y, './output_1_1024.csv')

if __name__ == '__main__':
    main(1)
