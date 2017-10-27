import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import regularizers
import random

def load_data():

    x = pd.read_csv('feature/X_train')
    x_norm = (x - x.mean()) / (x.std())
    X = (np.array(x_norm.values))

    y = pd.read_csv('feature/Y_train')
    y = (np.array(y.values))

    random.seed(8)
    random_l = random.sample(range(len(X)), len(X))

    print (len(random_l))

    X_train = []
    X_test = []
    cnt = 0
    for i in random_l:
        if (cnt / len(X)) < 0.8:
            X_train.append(X[i])
        else:
            X_test.append(X[i])
        cnt += 1

    print (len(X_train))
    print (len(X_test))


    y_train = []
    y_test = []
    cnt = 0
    for i in random_l:
        if (cnt / len(y)) < 0.8:
            y_train.append(y[i])
        else:
            y_test.append(y[i])
        cnt += 1

    print (len(y_train))
    print (len(y_test))



    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()
print (len(X_train[0]))
print (X_train[0])
print (X_train[1])
print (X_test[0])
print (X_test[1])
print (y_train[0])
print (y_train[1])
print (y_test[0])
print (y_test[1])

def train():
    model = Sequential()
    model.add(Dense(input_dim=len(X_train[0]), units=53))
    model.add(BatchNormalization())
    model.add(Dense(input_dim=len(X_train[0]), units=53, activation='relu'))
    #model.add(Dropout(0.7))
    model.add(Dense(kernel_initializer='normal', units=2, activation='softmax'))
    model.summary()

    #X_train = X_train /

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=106, epochs=10)
    score = model.evaluate(X_test, y_test)
    print ('\nTest Acc: ', score[1])

    return model

def predict(model):
    f = pd.read_csv('feature/X_test')

    test_x = f.values
    predict = model.predict(test_x)

    print (predict)

    pre = np.array([[0] if x[0] == 0 else [1] for x in predict])
    pre = pre.reshape(pre.shape[0], 1)
    print (pre.shape)

    index = np.array([[i] for i in range(1, test_x.shape[0] + 1)])
    print (index.shape)
    out_np = np.concatenate((index, pre), axis=1)
    out_df = pd.DataFrame(data=out_np, columns=['id', 'label'])
    out_df.to_csv('out_csv', index=False)

model = train()
predict(model)
