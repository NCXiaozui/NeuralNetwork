#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import random 


# Active function of Sigmod
class SigmodActivator():
    def forward(self, weighted_input):
        result = 1.0 / (1.0 + np.exp(-weighted_input))
        return result

    def backward(self, output):
        return output * (1 - output)

#
class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_szie = output_size
        self.activator = activator
        #init W matrix, row = output_size, raw = input_size 
        self.W = np.random.uniform(-0.1,0.1,(output_size,input_size))
        self.b = np.zeros((output_size,1))
        self.output = np.zeros((output_size,1))
        print input_size, output_size

    def output_result(self, input_result):
        self.input = input_result
        self.output = self.activator.forward(np.dot(self.W,self.input) + self.b)
        return self.output

    def backward_propagate(self, delta_arr):
        #因为更新是用未更新的参数，所以这一步要先计算下一次更新的参数
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_arr)
        self.W_grad =  delta_arr * self.input.T
        self.b_grad = delta_arr

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

class Network():
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i],layers[i+1],SigmodActivator()))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.output_result(output)
            output = layer.output
        return output

    def grandient(self, label):
        delta_arr = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward_propagate(delta_arr)
            delta_arr = layer.delta
        return delta_arr

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def train_one_sample(self, data, label, rate):
        self.predict(data)
        self.grandient(label)
        self.update_weight(rate)

    def train(self, data_set, label, rate, epoch):
        for i in range(epoch):
            for th in range(len(data_set)):
                self.train_one_sample(data_set[th], label[th], rate)
            



