from mlxtend.data import loadlocal_mnist
import random
import re
import math
import csv
import sys
import numpy as np
import pandas as pd
from Layer import Layer
from activation import sigmoid, tanh
from dense import Dense
from Neuron import Neuron

class LSTMClassifier:
    def __init__(self):
        self.n_layer = 0
        self.layers = []
        self.inputValue = []
        self.target = []
        self.dataTest = []

    def getLayer(self,idx):
        return self.layers[idx]

    def getInputModel(self):
        return self.inputValue
    
    # Fungsi buat convert
    def convertInt(self, arr): 
        numlist = [0 for i in arr]
        for i in range(len(arr)):
            numlist[i] = int(arr[i])
        return numlist

    # Fungsi buat convert
    def convertFloat(self, arr):
        numlist = [0 for i in arr]
        for i in range(len(arr)):
            numlist[i] = float(arr[i])
        return numlist

    def transposeMatrix(self, arr):
        result = []

        for i in range(len(arr[0])):
            result.append([0 for j in range(len(arr))])

        # iterate through rows
        for i in range(len(arr)):
            # iterate through columns
            for j in range(len(arr[0])):
                result[j][i] = arr[i][j]
        
        return result
    
    def loadCSV(self, train_path, test_path):

        # Read Data Training
        arrinput = []
        with open(train_path) as f:
            reader = csv.reader(f)
            headings = next(reader)
            for row in reader:
                isAppend = True
                for elm in row:
                    if elm == '-' or elm == None :
                        isAppend = False
                        break
                if isAppend :
                    arrRow = row[1:]
                    arrRow[-2] = arrRow[-2].replace(',', '')
                    arrRow[-1] = arrRow[-1].replace(',', '')
                    arrinput.append(arrRow)
        for i in range(len(arrinput)):
            arrinput[i] = self.convertFloat(arrinput[i])
        self.inputValue = arrinput

        # Read Data Test
        arrinput = []
        with open(test_path) as f:
            reader = csv.reader(f)
            headings = next(reader)
            for row in reader:
                arrRow = row[1:]
                arrRow[-2] = arrRow[-2].replace(',', '')
                arrRow[-1] = arrRow[-1].replace(',', '')
                arrinput.append(arrRow)
        for i in range(len(arrinput)):
            arrinput[i] = self.convertFloat(arrinput[i])
        self.dataTest = arrinput

    def load(self,filename):
        f = open(filename, "r")
        numbers = re.split('\n',f.read())
        # Jumlah layer
        self.n_layer = int(numbers[0])
        # Jumlah neuron per layer
        neurons = self.convertInt(re.split(' ',numbers[1]))

        layers = []
        for i in range(self.n_layer):
            isHiddenLayer = False
            if i > 0 or i < self.n_layer - 1:
                isHiddenLayer = True
            n_unit = neurons[i]
            n_unit_next = neurons[i]
            if i < self.n_layer - 1 :
                n_unit_next = neurons[i+1]
            newLayer = Layer(i,n_unit, n_unit_next, isHiddenLayer)
            layers.append(newLayer)
        self.layers = layers

    def feedFoward(self):
        print("Forward")

        ctValue = [0]
        htValue = [0]

        for x in self.inputValue:
            self.layers[0].setNeurons(x)
            for i in range(1, self.n_layer - 2):
                # Forget Gate
                ft = self.forget_gate(self.layers[i].getBackwardWeightsForgetGate(), self.layers[i-1].getNeurons(), self.layers[i].getWeightsForgetGate(), htValue, self.layers[i-1].getBias())
                # Input Gate
                it = self.forget_gate(self.layers[i].getBackwardWeightsInputGate(), self.layers[i-1].getNeurons(), self.layers[i].getWeightsInputGate(), htValue, self.layers[i-1].getBias())
                sigmaValue = np.dot(self.layers[i].getBackwardWeightsCellGate(), htValue) + np.dot(self.layers[i].getWeightsInputGate(), self.layers[i-1].getNeurons()) + self.layers[i-1].getBias()
                c_t = tanh(sigmaValue)
                # Cell Gate
                ct = np.dot(ft, ctValue) + np.dot(it, c_t)
                # Output Gate
                ot = self.forget_gate(self.layers[i].getBackwardWeightsOutputGate(), self.layers[i-1].getNeurons(), self.layers[i].getWeightsOutputGate(), htValue, self.layers[i-1].getBias())
                ht = np.dot(ot, tanh(ct))

                ctValue = ct
                htValue = ht

        # n_features = 6
        # n_unit_lstm = 1
        # n_time_step = 1

        # wf = np.random.random_sample((n_features,))
        # uf = np.random.random_sample((n_unit_lstm, ))
        # bf = np.random.random_sample((n_unit_lstm))
        
        # prev_h = [0]
        # for xt in self.inputValue:
        #     ft = self.forget_gate(wf, xt, uf, prev_h, bf)

    '''
        Fungsi menghitung forget gate ke-t
        Wf      : array of weight 
        x       : array of input x(t)
        Uf      : array of weight
        prev_h  : output h(t-1)
        bf      : array of bias
    '''
    def forget_gate(self, Wf, x, Uf, prev_h, bf):
        return sigmoid(np.dot(Wf, prev_h) + np.dot(Uf, x) + bf)

LSTM = LSTMClassifier()

LSTM.load("text.txt")
LSTM.loadCSV('bitcoin_price_training.csv', 'bitcoin_price_test.csv')
LSTM.feedFoward()