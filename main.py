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

        if np.isscalar(arr[0]):
            for j in range(len(arr)):
                result.append([arr[j]])
        else:
            for i in range(len(arr[0])):
                result.append([0 for j in range(len(arr))])
            # iterate through rows
            for i in range(len(result)):
                # iterate through columns
                for j in range(len(result[0])):
                    result[j][i] = arr[i][j]
        print(result)
        
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
            if i > 0 and i < self.n_layer - 1:
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
            print(x)
            self.layers[0].setNeurons(x)
            for i in range(1, self.n_layer - 1):
                # Forget Gate
                ft = self.forget_gate(self.layers[i-1].getWeightsForgetGate(), self.layers[i-1].getNeurons(), self.layers[i].getBackwardWeightsForgetGate(), htValue, self.layers[i-1].getBias())
                # print("Forget Gate ", ft)
                # Input Gate
                it = self.forget_gate(self.layers[i-1].getWeightsInputGate(), self.layers[i-1].getNeurons(), self.layers[i].getBackwardWeightsInputGate(), htValue, self.layers[i-1].getBias())
                # print("Input Gate ", it)
                sigmaValue = np.dot(self.layers[i].getBackwardWeightsCellGate(), htValue) + np.dot(self.layers[i-1].getWeightsInputGate(), self.layers[i-1].getNeurons()) + self.layers[i-1].getBias()
                # print("Nilai Sigma ", sigmaValue)
                c_t = tanh(sigmaValue)
                # print("Input Gate C_t ", c_t)
                # Cell Gate
                ct = np.dot(ft, ctValue) + np.dot(it, c_t)
                # print("Cell Gate ", ct)
                # Output Gate
                ot = self.forget_gate(self.layers[i-1].getWeightsOutputGate(), self.layers[i-1].getNeurons(), self.layers[i].getBackwardWeightsOutputGate(), htValue, self.layers[i-1].getBias())
                print("Output Gate ", ot)
                ht = np.dot(ot, tanh([ct]))
                print("Output Gate ht ", [ht])
                print("Output Cell Gate ", [ct])

                ctValue = [ct]
                htValue = [ht]
                self.layers[i].setNeurons(ot)

            # Dense Layer
            # denseLayer = Dense(1, 1)
            # output = self.layers[self.n_layer_konvolusi + 2].compute_output(ot)
            # print('OUTPUT : ', output)

    '''
        Fungsi menghitung forget gate ke-t
        Wf      : array of weight 
        x       : array of input x(t)
        Uf      : array of weight
        prev_h  : output h(t-1)
        bf      : array of bias
    '''
    def forget_gate(self, Uf, x, Wf, prev_h, bf):
        return sigmoid(np.dot(Uf, self.transposeMatrix(x)) + np.dot(Wf, prev_h) + bf)

    def summary(self):
        print("SUMMARY")
        params = 0
        for i in range(len(self.layers) - 1):
            if(i == len(self.layers) - 2):
                param = len(self.layers[i].getNeurons())
            else:
                param = (len(self.layers[i].getNeurons())) * (len(self.layers[i+1].getNeurons()) - 1)
            print("==================================")
            print("Layer (Type)    : dense_" + str(i) +" (Dense)")
            print("Param           : " +str(param))
            print("Output          : (None,"+ str(len(self.layers[i+1].getNeurons()))+")")
            print("Weight          :")
            params += param
        print("==================================")
        print("Total params   : " +str(params))

LSTM = LSTMClassifier()

LSTM.load("text.txt")
LSTM.loadCSV('bitcoin_price_training.csv', 'bitcoin_price_test.csv')
LSTM.feedFoward()
LSTM.summary()