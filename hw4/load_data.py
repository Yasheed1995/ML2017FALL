__author__ = 'b04901025'
# File script.py
import os
import sys
import numpy as np
import keras
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import h5py

VALIDATION_SPLIT = 0.2
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250

if __name__ == '__main__':
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

    print(len(X_train))
    print(len(y_train))
    print(len(X_valid))
    print(len(y_valid))
    with h5py.File('data/data.h5', 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_valid', data=X_valid)
        hf.create_dataset('y_valid', data=y_valid)
        hf.close()
