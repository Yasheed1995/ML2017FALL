import pandas as pd
import numpy as np
from keras.models import load_model
import argparse


def predict(test_path, model_path, csv_path):
    f = pd.read_csv(test_path)

    test_x = f['feature'].values
    test_x = (np.array([list(map(int, x.split(' '))) for x in test_x]))
    test_x = np.array([x.reshape((48,48)) for x in test_x]).reshape(-1,48,48,1)
    test_x = np.true_divide(test_x, 255)
    
    model = load_model(model_path)

    predict = model.predict(test_x)

    to_csv(predict, csv_path)

    #np.save('save/npy/test_y_' + model_name + '.npy', predict)

def to_csv(predict, csv_path):

    pre = []

    for x in predict:
        for n in range(7):
            if max(x) == x[n]:
                pre.append(n)

    pre = np.array(pre)

    pre = pre.reshape(pre.shape[0], 1)
    print (pre.shape)

    index = np.array([[i] for i in range(7178)])
    print (index.shape)
    out_np = np.concatenate((index, pre), axis=1)
    out_df = pd.DataFrame(data=out_np, columns=['id', 'label'])
    
    out_df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('--model_path', type=str, default='save/model/0.6660.hdf5')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--mode', type=str, default='predict')
    parser.add_argument('--csv_path', type=str, default='result/out.csv')
    parser.add_argument('--npy_name', type=str, default='')
    args = parser.parse_args()

    if args.mode == 'predict':
        predict(args.test_path, args.model_path, args.csv_path)




if __name__ == '__main__':
    main()
