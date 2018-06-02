# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:24:40 2018

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:24:13 2018

@author: ASUS
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DeepNeuralNet():
    def __init__(self, X, Y, classes):
        self.x_size = X.shape[1]
        self.y_size = Y.shape[1]
        self.params = {}
        self.gradients = {}
        self.caches = []
        self.classes = classes

        self.X = X
        self.L = 0
        self.Y = Y
        self.X_backup = np.copy(X)
        self.Y_backup = np.copy(Y)

    def init_deep_params(self, layer_dims):
        self.L = len(layer_dims)
#        print(self.L)
        for l in range(1, self.L):
            self.params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
#            print(self.params['W' + str(l)])
            self.params['b' + str(l)] = np.zeros((layer_dims[l], 1))
#            print(self.params['b' + str(l)])
#        print(self.params)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def diff_sigmoid(self, A):
        return A * (1 - A)

    def lin_forward(self, A_prev, W, b):

        Z = np.dot(W, A_prev) + b
        A = self.sigmoid(Z)
#        print("FF: ",A.shape)
        cache = (A_prev, W, b, A)

        return np.copy(A), cache

    def feed_forward(self):
        A = np.copy(self.X)
        h_L = len(self.params)//2
        for l in range(1, h_L):
            A_prev = A
            A , cache = self.lin_forward(A_prev, self.params['W' + str(l)], self.params['b'+str(l)])
#            print('W'+str(l))
            self.caches.append(cache)
        AL, cache = self.lin_forward(A, self.params['W'+str(h_L)], self.params['b'+str(h_L)])
        self.caches.append(cache)
#        print(self.params)
#        print("AL: ",AL.shape)
#        print(AL)
        return np.copy(AL)

    def cost_func(self, AL):
#        cost = -np.mean(np.multiply(np.log(AL),self.Y)+np.multiply(np.log(1-AL),1-self.Y))
#        cost = 0.5*np.dot((self.Y - AL), (self.Y - AL).T)/self.x_size
        cost = 0
        for i in range(self.classes):
#            print(self.Y[i])
            temp = 0.5*np.dot((self.Y[i] - AL[i]), (self.Y[i] - AL[i]).T)/self.x_size
            cost += np.squeeze(temp)
        return cost

    def lin_backward(self, dA, cache):
        A_prev, W, b, A = cache
        size = A_prev.shape[1]
#        print("dA: ",dA.shape)
#        print("A: ",A.shape)
        dZ = dA*np.copy(self.diff_sigmoid(A))
        dW = (1/size)*np.dot(dZ, A_prev.T)
        db = (1/size)*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev, dW, db

    def back_prop(self, AL):
        h_L = len(self.caches)
#        dAL = np.divide(1 - self.Y, 1 - AL) - np.divide(self.Y, AL)
#        length = self.Y.shape[1]
        dAL = AL - self.Y
#        print("caches size: ",h_L)
#        print("dAL: ",dAL.shape)
#        print("dAL: ",dAL)

        current = self.caches[h_L-1]
        self.gradients["dA" + str(h_L-1)], self.gradients["dW" + str(h_L)], self.gradients["db" + str(h_L)] = self.lin_backward(dAL, current)

        for l in reversed(range(h_L-1)):

            current = self.caches[l]
#            !print("l: ",l)
            dA_prev, dW, db = self.lin_backward(self.gradients["dA"+str(l+1)], current)
            self.gradients["dA" + str(l)] = dA_prev
            self.gradients["dW" + str(l + 1)] = dW
            self.gradients["db" + str(l + 1)] = db

    def update(self, learning_rate):
        h_L = len(self.params)//2
        for l in range(h_L):
            self.params['W'+str(l+1)] = self.params['W'+str(l+1)] - learning_rate*self.gradients['dW'+str(l+1)]
            self.params['b'+str(l+1)] = self.params['b'+str(l+1)] - learning_rate*self.gradients['db'+str(l+1)]

    def train(self, layer_dims, epochs, learning_rate):
        costs = []

        self.init_deep_params(layer_dims)
#        batch = 1
#        total = self.X_backup.shape[1]
        for i in range(epochs):
#            for j in range(total):
#                print("batch: ",j+1)
#                start = j*batch
#                last = j*batch+batch
#                self.X = self.X_backup[:,start:last].copy()
#                self.Y = self.Y_backup[:, start:last].copy()
#                sel1f.x_size = self.X.shape[1]
            AL = self.feed_forward()
#            print("epoch ",i," AL: ",AL.shape)
            cost = self.cost_func(AL)
            self.back_prop(AL)
            self.update(learning_rate)
            self.init_cache()
            costs.append(cost)
            print('epoch ',i,': cost = ',cost)

    def predict(self, AL, y_test):
        miss = 0
#        l = AL.T
#        target = self.Y.T
        correct = 0
        maxi = np.argmax(AL, axis=0)
#        print(AL)
#        print(y_test)
#        print(maxi)
#        for j in range(len(y_test)):
        if maxi[0] != y_test-1:
            miss += 1
        else:
            correct += 1
        return correct, miss


    def init_cache(self):
        self.caches = []



    def test(self, x_test, y_test, y_test_feed):
        self.X_backup = x_test
        self.Y_backup = y_test_feed
#        self.x_size = x_test.shape[1]
        total = self.X_backup.shape[1]
        batch = 1
        correct =0
        miss =0
        for j in range(total):
#            print("batch: ",j+1)
            start = j*batch
            last = j*batch+batch
            self.X = self.X_backup[:,start:last].copy()
            self.Y = self.Y_backup[:, start:last].copy()
            self.x_size = self.X.shape[1]
            AL = self.feed_forward()

            c, m = self.predict(AL, y_test[j])
            correct += c
            miss += m
        print("correct: ",correct)
        print("miss: ",miss)




def preprocessY(Y):
    classes = np.unique(Y)
    count = len(classes)
    length = Y.shape[1]
#    print(count)
#    print(classes)

    class_dict = {}
    index = 0
    for cls in classes:
        class_dict[cls] = index
        index+=1
#    print(class_dict)
    new_Y = np.zeros((count, length))
#    print(new_Y)
    for i in range(length):
        idx = class_dict[Y[0][i]]
#        print(idx)
        new_Y[idx][i] = 1
#        print(each)
#    print(new_Y)
    return new_Y, count







def main():
    np.random.seed(10)
    filename = "F:/Class materials/L4T2/PR/offline1/1305063/train.csv"
    test = "F:/Class materials/L4T2/PR/offline1/1305063/test.csv"
    data = pd.read_csv(filename, sep=",")
    data_test = pd.read_csv(test, sep=",")
#    print(len(data.loc[0]))
#    print(data.loc[1])
#    y_data = data['col8']
    data = shuffle(data, random_state=20)
    data_columns = data.columns.values
    data_test = shuffle(data_test, random_state=20)
    data_columns_test = data_test.columns.values
#    print(data_columns[-1])
    l_size = len(data_columns)
    Y = data.as_matrix([data_columns[l_size-1]])
    X = data.as_matrix(data_columns[:l_size-1])

    l_size_test = len(data_columns_test)
    testY = data_test.as_matrix([data_columns_test[l_size_test-1]])
    testX = data_test.as_matrix(data_columns_test[:l_size_test-1])
#    print(Y)
#    x_data = data['col1','col8']
#    print(Y)
#    x_train1, x_test, y_train1, y_test = train_test_split(X, Y, test_size=.20, random_state=10, shuffle=True)



    x_train = X.T
#    print(x_train)
    y_train = Y.T
    y_train, classes = preprocessY(y_train)
    print("classes: ",classes)

#    length = y_test.shape[0]
#    x_test = x_test.T
#    y_test = y_test.values.reshape((length, 1)).T

#    print(x_train.shape)
#    print(y_train.shape)
#    print(x_test.shape)
#    print(y_test.shape)
    start = time.clock()
    x_test = testX.T
    y_test = testY.T
    y_test_feed, count = preprocessY(y_test)

    dnn = DeepNeuralNet(x_train, y_train, classes)
#    dnn.init_deep_params([7, 3, 4, 3])
    dnn.train([3, 3, 3, 3], 15000, 0.5)
    y_test = list(np.ndarray.flatten(y_test))
#    print("y_test_feed:")
#    print(y_test_feed)
    dnn.test(x_test, y_test, y_test_feed)
    end = time.clock()
    print(end-start,"s")



if __name__ == '__main__':
    main()