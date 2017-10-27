import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

def load_data():
    a = 32000
    X_train = pd.read_csv('feature/X_train')
    X_train = np.array(X_train)
    y_train = pd.read_csv('feature/y_train')
    y_train = np.array(y_train)

    X_test  = pd.read_csv('feature/X_test')
    X_test  = np.array(X_test)
    idx     = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    X_valid = np.array(X_train[a:, :])
    y_valid = np.array(y_train[a:, :]).ravel()

    X_train = np.array(X_train[:a, :])
    y_train = np.array(y_train[:a, :]).ravel()


    return (X_train, y_train), (X_valid, y_valid), (X_test)

def write(out, filename):
    with open(filename, 'w') as output:
        output.write('id,label')
        count = 1
        for i in out:
            output.write('\n')
            output.write(str(count))
            output.write(',')
            output.write(str(int(i)))
            count += 1


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test) = load_data()
    # normalize
    X_train = (X_train - np.min(X_train, axis=0)) / np.max(X_train, axis=0)
    X_valid = (X_valid - np.min(X_train, axis=0)) / np.max(X_train, axis=0)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    print (sum(np.round(model.predict(X_valid)) == y_valid) / float(len(y_valid)))

    write(np.rint(model.predict(X_test)), 'answer.csv')

if __name__ == '__main__':
    main()


