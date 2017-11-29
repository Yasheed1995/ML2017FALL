# -*- coding: utf-8 -*-
__author__ = 'b04901025'
# File train.py
import tensorflow   as tf
import numpy        as np
import os
import argparse
import keras
import h5py
import os
import sys
import numpy as np
import keras
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import h5py
from keras.models               import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers           import Adam, Adadelta
from keras.utils                import np_utils
from keras                      import regularizers
from keras.preprocessing.image  import ImageDataGenerator
from keras.layers               import Dense, Dropout, Activation, Flatten
from keras.layers               import Convolution1D, Conv1D, AveragePooling1D
from keras.layers               import MaxPooling1D
from keras.layers               import LSTM
from keras.layers.embeddings    import Embedding
from keras.preprocessing        import sequence
from keras.callbacks            import ModelCheckpoint,EarlyStopping

MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def main():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='save/model/model-1')
    parser.add_argument('--which_model', type=int, default=1)
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--gpu_fraction', type=float, default=0.9)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.which_gpu)

    # gpu limitation
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))

    (X_train, y_train), (X_valid, y_valid), embedding_layer = load_data()

    train(
        args.which_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        args.batch,
        args.epoch,
        args.pretrain,
        args.model_name,
        embedding_layer
    )

def prepare_embedding(word_index):
    embeddings_index = {}
    files = []
    files.append(open('data/glove.6B/glove1.txt', 'r'))
    files.append(open('data/glove.6B/glove2.txt', 'r'))
    files.append(open('data/glove.6B/glove3.txt', 'r'))
    files.append(open('data/glove.6B/glove4.txt', 'r'))

    for f in files:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#print('Training model.')
    return embedding_layer
def load_data():
        buffer_ = []
        texts = []
        labels = []
        texts_labels = {}
        with open('data/training_label.txt', 'r') as f:
            buffer_ = f.read()
            print(len(buffer_.split('\n')))
            for line in buffer_.split('\n'):
                if line == "":
                    break
                texts.append(line[10:])
                labels.append(line[0])
            #print np.array(texts)
            #print np.array(labels)
        tokenizer = Tokenizer(MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        labels = np_utils.to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        X_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        X_valid = data[-nb_validation_samples:]
        y_valid = labels[-nb_validation_samples:]

        embedding_layer = prepare_embedding(word_index)

        return (X_train, y_train), (X_valid, y_valid), embedding_layer

def train(which_model, X_train, y_train, X_valid, y_valid,
         n_batch, n_epoch, pretrain, model_name, embedding_layer):

    if pretrain == False:
        if which_model == 0:
            model = build_model_0(embedding_layer)
        elif which_model == 1:
            model = build_model_1(embedding_layer)
        #elif which_model == 2:
        #    model = build_model_2()
    else:
        model = load_model(model_name)

    filepath='save/Model.{epoch:02d}-{val_acc:.4f}.hdf5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
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

    history = model.fit(X_train, y_train, batch_size=n_batch, epochs=n_epoch,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks_list)

    model.save('save/model/' + str(model_name) + '.hdf5')


    score = model.evaluate(X_valid, y_valid)
    print ('\nTest Acc:', score[1])

def build_model_0(embedding_layer):
    model = Sequential()
    #embedding_layer = prepare_embedding(word_index)
    model.add(embedding_layer)
    #model.add(Embedding(160000, 64, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def build_model_1(embedding_layer):
    model = Sequential()
    #embedding_layer = prepare_embedding(word_index)
    model.add(embedding_layer)
    model.add(LSTM(100))
    #model.add(Conv1D(128, 5, activation='sigmoid'))
    #model.add(MaxPooling1D(5))
    #model.add(Conv1D(128, 5, activation='sigmoid'))

    #model.add(MaxPooling1D(5))
    #model.add(Conv1D(128, 5, activation='sigmoid'))
    #model.add(MaxPooling1D(1))

    #model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))


    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    main()



