import pandas as pd
import numpy as np
from numpy.random import permutation
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add, Concatenate
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add
assert Model
assert np_utils and permutation

class DataManager:
    def __init__(self):
        self.data = {}
        self.id = {}

    def read_data(self, path, name, with_label=True, read_id=False):

        if with_label:
            f = pd.read_csv(path)
            uid = np.array(f.iloc[:]['UserID']).reshape(-1,1)
            mid = np.array(f.iloc[:]['MovieID']).reshape(-1,1)
            rat = np.array(f.iloc[:]['Rating']).reshape(-1,1)
            #self.Data_normalization(rat)
            data = permutation(np.hstack((uid, mid, rat)))
            X = data[:, 0:2]
            Y = data[:, 2].reshape(-1,1)
            self.data[name] = [X, Y]

        elif not read_id:
            f = pd.read_csv(path)
            uid = np.array(f.iloc[:]['UserID']).reshape(-1,1)
            mid = np.array(f.iloc[:]['MovieID']).reshape(-1,1)
            X = np.hstack((uid, mid))
            self.data[name] = [X]

        else:
            ff = open(path, 'r', encoding='latin-1')
            id_n=[]
            next(ff)
            for line in ff:
                id_n.append(int(line.split('::', 1)[0]))
            maxid = max(id_n)
            self.id[name] = [id_n, maxid]
            ff.close()

    def Data_normalization(self, arr): # normalize train rating
        self.train_rating_std = arr.std()
        self.train_rating_mean = arr.mean()
        arr = ((arr - arr.mean()) / arr.std())


    def build_model_0(self):
        u_input = Input(shape=(1,), name='user_input')
        m_input = Input(shape=(1,), name='movie_input')

        e1_u = Flatten()(Embedding(output_dim=128, input_dim=self.id['user'][1],  input_length=1)(u_input))
        e2_u = Flatten()(Embedding(output_dim=1,  input_dim=self.id['user'][1],  input_length=1)(u_input))
        e1_m = Flatten()(Embedding(output_dim=128, input_dim=self.id['movie'][1], input_length=1)(m_input))
        e2_m = Flatten()(Embedding(output_dim=1,  input_dim=self.id['movie'][1], input_length=1)(m_input))


        output = Add(name='output')([e2_u, e2_m, Dot(1)([e1_m, e1_u])])
        model = Model(inputs=[u_input, m_input], outputs=[output])
        #model.compile('adam', 'categorical_crossentropy')
        self.model_0 = model

    def build_model_TA(self, n_users=6040, n_items=3883, latent_dim=1024):
        user_input = Input(shape=[1], name='user_input')
        item_input = Input(shape=[1], name='movie_input')
        user_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(user_input)
        user_vec = Flatten()(user_vec)
        item_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(item_input)
        item_vec = Flatten()(item_vec)
        user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
        user_bias = Flatten()(user_bias)
        item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
        item_bias = Flatten()(item_bias)
        r_hat = Dot(axes=1)([user_vec, item_vec])
        r_hat = Add(name='output')([r_hat, user_bias, item_bias])
        model = Model([user_input, item_input], r_hat)
        self.model_1 = model
        # get embedding
        user_emb = np.array(model.layers[2].get_weights()).squeeze()
        movie_emb = np.array(model.layers[3].get_weights()).squeeze()
        print ('user embedding shape:', user_emb.shape)
        print ('movie embedding shape:', movie_emb.shape)
        np.save('save/user_emb_1.npy', user_emb)
        np.save('save/movie_emb_1.npy', movie_emb)

    def build_model_TA_no_bias(self, n_users=6040, n_items=3883, latent_dim=1024):
        user_input = Input(shape=[1], name='user_input')
        item_input = Input(shape=[1], name='movie_input')
        user_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(user_input)
        user_vec = Flatten()(user_vec)
        item_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(item_input)
        item_vec = Flatten()(item_vec)
        #user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
        #user_bias = Flatten()(user_bias)
        #item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
        #item_bias = Flatten()(item_bias)
        r_hat = Dot(axes=1, name='output')([user_vec, item_vec])
        #r_hat = Add(name='output')([r_hat, user_bias, item_bias])
        model = Model([user_input, item_input], r_hat)
        self.model_no_bias = model
        # get embedding
        user_emb = np.array(model.layers[2].get_weights()).squeeze()
        movie_emb = np.array(model.layers[3].get_weights()).squeeze()
        print ('user embedding shape:', user_emb.shape)
        print ('movie embedding shape:', movie_emb.shape)
        np.save('save/user_emb_1.npy', user_emb)
        np.save('save/movie_emb_1.npy', movie_emb)

    def build_model_TA_DNN(self, n_users=6040, n_items=3883, latent_dim=1024):
        user_input = Input(shape=[1], name='user_input')
        item_input = Input(shape=[1], name='movie_input')
        user_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(user_input)
        user_vec = Flatten()(user_vec)
        item_vec = Embedding(n_users, latent_dim,
                             embeddings_initializer='random_normal')(item_input)
        item_vec = Flatten()(item_vec)
        merge_vec = Concatenate()([user_vec, item_vec])
        hidden = Dense(128, activation='relu')(merge_vec)
        hidden = Dense(64, activation='relu')(hidden)
        output = Dense(1, name='output')(hidden)
        model = Model([user_input, item_input], output)
        self.model_dnn = model
        # get embedding
        user_emb = np.array(model.layers[2].get_weights()).squeeze()
        movie_emb = np.array(model.layers[3].get_weights()).squeeze()
        print ('user embedding shape:', user_emb.shape)
        print ('movie embedding shape:', movie_emb.shape)
        np.save('save/user_emb_2.npy', user_emb)
        np.save('save/movie_emb_2.npy', movie_emb)

    def MSE(self, y_true, y_pred):
        return (K.mean(K.square(y_pred - y_true), axis=-1))

    def write_file(self, test, path):
        idx = np.array([[j for j in range(1, len(test)+1)]]).T
        test = np.hstack((idx, test))

        out = pd.DataFrame(test, columns=['TestDataID', 'Rating'])
        out['TestDataID'] = out['TestDataID'].astype(int)
        out.to_csv(path, index=False)
