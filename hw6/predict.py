import os
import tensorflow as tf
import sys 
sys.path.append("./")
import numpy as np
import keras
assert np and os and tf and set_session

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster  import SpectralClustering , KMeans
import pandas as pd
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import sys
import matplotlib.pyplot as plt

n_batch=512
n_epoch=30

adam=keras.optimizers.Adam()
adamax=keras.optimizers.Adamax()


class   DataManager:
    def __init__(self):
        self.data={}
    def load_image(self,path,name):
        tmp=np.load(path)
        self.data[name]=(tmp-np.mean(tmp,axis=0))/255
    def load_test(self,path,name):
        x=pd.read_csv(path)
        self.data[name]=np.array(x.iloc[:,1:])
    def predict(self,data,name):
        #predict data index with the name of the data.
        res=[]
        one_c=0
        zero_c=0
        for i in self.data[name]:
            if data[i[0]]==data[i[1]]:
                res.append(1)
                one_c+=1
            else:
                res.append(0)
                zero_c+=1
        return np.array(res).reshape((-1,1))
    def write(self,test,path):
        #write data to path.
        idx=np.array([[j for j in range(len(test))]]).T
        test=np.hstack((idx,test))
        myoutput=pd.DataFrame(test,columns=["ID","Ans"])
        myoutput.to_csv(path,index=False)

dm =DataManager()
dm.load_image(sys.argv[1],'image')
dm.load_test(sys.argv[2],'test')

lib = np.load('lib.npy')
test_y=dm.predict(lib, 'test')
dm.write(test_y, sys.argv[3])
