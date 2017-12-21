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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

sys.path.append('./data')

n_batch=2048

def main(_model):

    dm = DataManager()
    dm.read_data('data/train.csv', 'train', with_label=True)
    dm.read_data('data/test.csv', 'test', with_label=False)
    dm.read_data('data/users.csv', 'user', with_label=False, read_id=True)
    dm.read_data('data/movies.csv', 'movie', with_label=False, read_id=True)

    dm.build_model_0()
    dm.build_model_TA()
    dm.build_model_TA_DNN()
    dm.build_model_TA_no_bias()
    if _model == 0:
        model = dm.model_0
    elif _model == 1:
        model = dm.model_1
    elif _model == 2:
        model = dm.model_dnn
    else:
        model = dm.model_no_bias
    adam = keras.optimizers.Adam(clipnorm=0.0001)
    model.compile(optimizer=adam,
                  loss={'output': 'mean_squared_error'},
                  metrics=['mean_squared_error'])

    filepath = 'model/model_1_1024.hdf5'
    checkpoint1 = ModelCheckpoint(filepath,
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min')
    chechpoint2 = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=3,
                                verbose=0,
                                mode='min')

    callbacks_list = [checkpoint1, chechpoint2]

    model.summary()

    history = model.fit({'user_input': dm.data['train'][0][:,0],
                         'movie_input': dm.data['train'][0][:,1]},
              {'output': dm.data['train'][1][:,0]},
             epochs=60,
             batch_size=n_batch,
             shuffle=True,
             validation_split=0.05,
             callbacks=callbacks_list,
             verbose=1)

    dict_history = pd.DataFrame(history.history)
    dict_history.to_csv('save/history_1024.csv')

    test_y = model.predict({'user_input':dm.data['test'][0][:,0],
                            'movie_input':dm.data['test'][0][:,1]},
                          batch_size=n_batch, verbose=1)
    # data normalization
    #test_y = test_y * dm.train_rating_std + dm.train_rating_mean
    np.save('test_y_1_1024.npy', test_y)
    dm.write_file(test_y, './output_1_1024.csv')

if __name__ == '__main__':
    main(1)
