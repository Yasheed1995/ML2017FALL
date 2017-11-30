__author__ = 'b04901025'

import os
import sys
import numpy as np
import keras
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import h5py

MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

import pandas as pd
import numpy as np
from keras.models import load_model
import argparse


def predict(test_path, model_path):
    buffer_ = []
    texts = []
    texts_labels = {}
    with open('data/training_label.txt', 'r') as f:
        buffer_ = f.read()
        print(len(buffer_.split('\n')))
        for line in buffer_.split('\n'):
            if line == "":
                break
            index_of_comma = line.find(',')
            texts.append(line[index_of_comma:])
    tokenizer = Tokenizer(MAX_NB_WORDS)
        
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
        
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    model = load_model(model_path)

    predict = model.predict(data)
    print (predict)

    #to_csv(predict, csv_path)

    np.save('save/npy/test_y_' + '0.7961' + '.npy', predict)

def to_csv(npy_path, csv_path):
    
    predict = np.load(npy_path)
    pre = []

    for x in predict:
        for n in range(2):
            if max(x) == x[n]:
                pre.append(n)

    pre = np.array(pre)

    pre = pre.reshape(pre.shape[0], 1)
    print (pre.shape)

    index = np.array([[i] for i in range(200000)])
    print (index.shape)
    out_np = np.concatenate((index, pre), axis=1)
    out_df = pd.DataFrame(data=out_np, columns=['id', 'label'])

    out_df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('--model_path', type=str, default='save/Model.07-0.7961.hdf5')
    parser.add_argument('--test_path', type=str, default='data/testing_data.txt')
    parser.add_argument('--mode', type=str, default='predict')
    parser.add_argument('--csv_path', type=str, default='result/out.csv')
    parser.add_argument('--npy_path', type=str, default='save/npy/test_y_0.7975.npy')
    args = parser.parse_args()

    if args.mode == 'predict':
        predict(args.test_path, args.model_path)
    elif args.mode == 'csv':
        to_csv(args.npy_path, args.csv_path)




if __name__ == '__main__':
    main()

