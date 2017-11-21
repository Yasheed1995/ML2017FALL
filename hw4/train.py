# -*- coding: utf-8 -*-
__author__ = 'b04901025'
# File train.py
import tensorflow   as tf
import numpy        as np
import os
import argparse
import keras
import h5py
from keras.models               import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers           import Adam, Adadelta
from keras.utils                import np_utils
from keras                      import regularizers
from keras.preprocessing.image  import ImageDataGenerator
from keras.layers               import Dense, Dropout, Activation, Flatten
from keras.layers               import Convolution2D, Conv2D, AveragePooling2D
from keras.layers               import ZeroPadding2D, MaxPooling2D
from keras.layers.embeddings    import Embedding
from keras.callbacks            import ModelCheckpoint,EarlyStopping

def main():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='save/model/model-1')
    parser.add_argument('--which_model', type=int, default=0)
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--gpu_fraction', type=float, default=1.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.which_gpu)

    # gpu limitation
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))

    (X_train, y_train), (X_valid, y_valid) = load_data()

    train(
        args.which_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        args.batch,
        args.epoch,
        args.pretrain,
        args.model_name
    )


def load_data():
    with h5py.File('data/data.h5', 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_valid = hf['X_valid'][:]
        y_valid = hf['y_valid'][:]

    return (X_train, y_train), (X_valid, y_valid)

def train(which_model, X_train, y_train, X_valid, y_valid,
         n_batch, n_epoch, pretrain, model_name):

    if pretrain == False:
        if which_model == 0:
            model = build_model_0()
        #elif which_model == 1:
        #    model = build_model_1()
        #elif which_model == 2:
        #    model = build_model_2()
    else:
        model = load_model(model_name)

    filepath='save/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    callbacks_list = [checkpoint1,checkpoint2]

    '''
    model.fit_generator(datagen_train.flow(X_train, y_train,
                    batch_size=n_batch),
                    epochs=n_epoch,
                    steps_per_epoch=len(X_train)/n_batch,
                    validation_data=datagen_test.flow(X_valid, y_valid,
                        batch_size=n_batch),
                    validation_steps=len(X_valid)/n_batch,
                    samples_per_epoch=X_train.shape[0],
                    callbacks=callbacks_list)
    '''

    model.fit(X_train, y_train, batch_size=n_batch, epochs=n_epoch, callbacks=callbacks_list)
    model.save('save/model/' + str(model_name) + '.hdf5')

    score = model.evaluate(X_valid, y_valid)
    print ('\nTest Acc:', score[1])

def build_model_0():
    model = Sequential()
    model.add(Embedding(160000, 64, input_length=1000))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['val_acc'])
    model.summary()

    return model

if __name__ == '__main__':
    main()

