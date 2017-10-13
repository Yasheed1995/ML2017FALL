# -*- coding: utf-8 -*-
__author__ = 'b04901025'
import numpy as np
from numpy.linalg import inv
import sys
import math
import csv

class Preprocess:
    def __init__(self, dataset, feature_normalization=True):
        self.dataset 				= dataset
        self.feature_normalization 	= feature_normalization

    def normalization(self, data):
        if self.feature_normalization is True:
            return [((x - np.mean(x)) / np.std(x)) for x in data]
        else:
            return data

    def data_prep(self, dataset_type):

        if dataset_type == "Train":
            X 		= []
            y 		= []
            data_ 	= []
            for i in range(18):
                data_.append([])

            row_count = 0
            row = csv.reader(self.dataset, delimiter=',')
            for r in row:
                if row_count != 0:
                    for i in range(3, 27):
                        if r[i] != 'NR':
                            data_[(row_count-1)%18].append(float(r[i]))
                        else:
                            data_[(row_count-1)%18].append(float(0))
                row_count += 1
            data = self.normalization(data_)

            for i in range(12):
                for j in range(471):
                    X.append([])
                    for t in range(8,10):
                        for s in range(9):
                            X[471*i+j].append(data[t][480*i+j+s])
                    y.append(data[9][480*i+j+9])
            X = np.array(X)
            y = np.array(y)

            return X, y

        else:
            test_x 		= []
            row_count 	= 0
            row = csv.reader(self.dataset, delimiter=',')

            for r in row:
                if row_count % 18 == 0:
                    test_x.append([])
                    #for i in range(2, 11):
                        #test_x[row_count//18].append(float(r[i]))
                elif row_count % 18 == 8 or row_count % 18 == 9:
                    for i in range(2, 11):
                        if r[i] != 'NR':
                            test_x[row_count//18].append(float(r[i]))
                        else:
                            test_x[row_count//18].append(float(0))

                row_count += 1
            test_x = np.array(test_x)

            return test_x

class LinearRegression:

    def __init__(self, X=None, y=None, iteration=10000, lr=10, lambda_r=1000, add_bias=False, add_square_term=False, regularization=False, theta=None, flag='Train'):
        self.X 					= X
        self.y 					= y
        self.add_bias			= add_bias
        self.add_square_term 	= add_square_term
        self.regularization 	= regularization
        self.iteration			= iteration
        self.lr					= lr # learn rate
        self.lambda_r       	= lambda_r
        self.flag               = flag

        if flag is 'Train':
            if self.add_bias is True:
                self.X 			= np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)

            if self.add_square_term is True:
                self.X 			= np.concatenate((self.X, self.X**2), axis=1)

            self.theta		    = np.random.uniform(low=(-0.2), high=(0.2), size=len(self.X[0]))
        else:
            self.theta          = theta # for testing

        # self.theta 			= np.zeros(len(self.X[0]))
    def Computecost(self, X, y, theta):
        prediction 				    = np.dot(X, theta)
        if self.flag is 'Train':
            loss                	= prediction - y
            error 					= np.power((prediction - y), 2)
            if self.regularization is True:
                cost 				= (np.sum(error) / len(X)) + self.lambda_r * np.dot(theta, theta)
            else:
                cost 				= (np.sum(error) / len(X))
            return prediction, loss, cost
        else:
            return prediction

    def gradient_descent(self):
        x_t 					= self.X.transpose()
        cost_list 				= []
        s_gra 					= np.zeros(len(self.X[0]))

        for i in range(self.iteration):
            # compute cost
            prediction, loss, cost 	= self.Computecost(self.X, self.y, self.theta)
            cost_sqrt 				= math.sqrt(cost)
            # compute gradient
            gra 					= (-2) * np.dot(x_t, loss)
            # adagrad
            s_gra 					+= gra ** 2
            ada 					= np.sqrt(s_gra)
            # update theta
            self.theta 				+= self.lr * gra / ada
            if(i%1000 == 0):
                print ("iteration: {0} | Cost: {1}".format(i, cost_sqrt))
            cost_list.append(cost_sqrt)
        return self.theta

    def predict_test(self, test_x):
        theta 					= self.theta
        ans 					= []
        ans_predicted           = []
        if(self.add_bias == True):
            test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

        if self.add_square_term is True:
            test_x = np.concatenate((test_x, test_x ** 2), axis=1)


        for i in range(len(test_x)): # 240
            ans.append(["id_" + str(i)])
            prediction = self.Computecost(test_x[i], self.y, theta)
            ans[i].append(prediction)
            ans_predicted.append(prediction)

        self.write_ans(ans)

        np.save('data/prediction.npy', np.array(ans_predicted))

    def write_ans(self, ans):
        filename 	= sys.argv[2]
        t 			= open(filename, 'w+')
        s 			= csv.writer(t, delimiter=',', lineterminator='\n')
        s.writerow(["id", "value"])
        for i in range(len(ans)):
            s.writerow(ans[i])
        t.close()

def train():
	training_data			= open(sys.argv[1], 'r', encoding='big5')

	dataset_type_train 		= "Train"

	add_bias 				= True
	feature_normalization 	= False
	add_square_term 		= True
	regularization 			= True

	iteration 				= 40001
	lr 						= 10
	lambda_r 				= 1000

	obj						= Preprocess(training_data, feature_normalization)
	x_train, y_train		= obj.data_prep(dataset_type_train)

    # save for close form sol
	np.save("data/x_train.npy", x_train)
	np.save("data/y_train.npy", y_train)

	obj1 					= LinearRegression(x_train, y_train, iteration, lr, lambda_r, add_bias, add_square_term)
	theta_trained			= obj1.gradient_descent()


	training_data.close()

	#print (predictedvalue)
	#print (x_test)
	return theta_trained

def test(theta_trained):
	test_data 				= open(sys.argv[1], 'r', encoding='big5')
	dataset_type_test		= "Test"

	add_bias 				= True
	feature_normalization 	= False
	add_square_term 		= True
	regularization 			= True

    # create a temp obj
	obj_temp = LinearRegression(theta=theta_trained, flag='Test', add_bias=add_bias, add_square_term=add_square_term, regularization=regularization)

	obj_test 				= Preprocess(test_data)
	x_test					= obj_test.data_prep(dataset_type_test)

    # for close from
	np.save('data/x_test.npy', x_test)

	predictedvalue			= obj_temp.predict_test(x_test)

	print ("test done")
	test_data.close()

if __name__ == '__main__':
	if sys.argv[1] == 'data/train.csv':
		theta_trained = train()
		np.save('data/model.npy', theta_trained)
	else: # test
		theta_trained = np.load('data/model.npy')
		test(theta_trained)
