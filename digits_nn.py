#!usr/bin//env python
# -*- coding: UTF-8 -*-

import numpy as np
import sklearn.datasets as skd
from bp import *
from sklearn.cross_validation import train_test_split
from datetime import datetime



def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = xtrain,ytrain
    test_data_set, test_labels = xtest,ytest
    network = Network([64, 200, 10])
    while True:
        epoch += 1
        network.train(train_data_set, train_labels, 0.2, 1)
        print '%s epoch %d finished' % (datetime.now(), epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print '%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
            print label, predict, '\n',network.predict(test_data_set[i])
    return float(error) / float(total)

if __name__ == '__main__':
    digits = skd.load_digits()
    x,y = digits['data'][:],digits['target'][:]
    x = map(lambda d: np.array(d).reshape(64,1),x)
    label = []
    for i in y:
        temp = np.zeros((10,1))
        temp[i] = 1.0
        label.append(temp)
    xtrain,xtest,ytrain,ytest = train_test_split(x,label,test_size=0.3,random_state=0)
    train_and_evaluate()