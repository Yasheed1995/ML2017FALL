# -*- coding: utf-8 -*-
__author__ = 'b04901025'
import tensorflow as tf
import numpy      as np
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
from keras.callbacks            import ModelCheckpoint,EarlyStopping
#CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# gpu limitation
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

def main():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='save/model/model-1')
    parser.add_argument('--which_model', type=int, default=0)
    args = parser.parse_args()

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
        elif which_model == 1:
            model = build_model_1()
        elif which_model == 2:
            model = build_model_2()
    else:
        model = load_model(model_name)

    datagen_train, datagen_test = data_gen(X_train, y_train, X_valid, y_valid)

    filepath='save/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    callbacks_list = [checkpoint1,checkpoint2]

    model.fit_generator(datagen_train.flow(X_train, y_train,
                    batch_size=n_batch),
                    epochs=n_epoch,
                    steps_per_epoch=len(X_train)/n_batch,
                    validation_data=datagen_test.flow(X_valid, y_valid,
                        batch_size=n_batch),
                    validation_steps=len(X_valid)/n_batch,
                    samples_per_epoch=X_train.shape[0],
                    callbacks=callbacks_list)
    model.save('save/model/' + str(model_name) + '.hdf5')

    score = model.evaluate(X_valid, y_valid)
    print ('\nTest Acc:', score[1])

def data_gen(X_train, y_train, X_valid, y_valid):

    for i in range(2):
        X_train = np.vstack((X_train, X_train))
        y_train = np.vstack((y_train, y_train))

    datagen_valid = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False)
    datagen_train = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    #datagen_train.fit(X_train)
    #datagen_valid.fit(X_valid)

    return datagen_train, datagen_valid

def build_model_0(): # model 1
    model = Sequential()
    model.add(Convolution2D(64,5,5,activation='relu',input_shape=(48,48,1)))
    for i in range(4):
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(Convolution2D(64,3,3,activation='relu'))
        model.add(BatchNormalization())
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(Convolution2D(64,3,3,activation='relu'))
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
    model.add(Flatten())

    for i in range(3):
        model.add(Dense(units=1024,activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout((i+13)*0.05))
    model.add(Dense(units=7,activation='softmax'))

    adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def build_model_1(): # model 2
    img_rows, img_cols = 48, 48
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))


    model.add(Dense(7))


    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model

def build_model_2():
    model = Sequential()
    model.add(Convolution2D(128,5,5,activation='relu',input_shape=(48,48,1)))
    for i in range(3):
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(Convolution2D(256,3,3,activation='relu'))
        model.add(BatchNormalization())
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(Convolution2D(256,3,3,activation='relu'))
        model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
    model.add(Flatten())

    for i in range(4):
        model.add(Dense(units=2**(10-i),activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout((8-i)*0.1))
    model.add(Dense(units=7,activation='softmax'))

    adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    main()

