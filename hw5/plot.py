#!/usr/bin/env python
__author__ = 'Yasheed'
# File script.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    f = pd.read_csv('save/' + sys.argv[1] + '.csv')

    loss = f['loss']
    val_loss = f['val_loss']
    print (loss.shape)
    x = np.arange(loss.shape[0])

    plt.title(sys.argv[1] + ' plot')
    plt.xlabel('# of epoch')
    plt.ylabel('loss')
    plt.yscale("log", nonposy='clip')
    plt.plot(x, loss,'k--', label='loss')
    plt.plot(x, val_loss, label='val_loss')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('image/' + sys.argv[1] + '_log.png')
    #plt.show()
