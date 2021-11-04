from mlxtend.data import loadlocal_mnist
import random
import re
import math
import csv
import sys
import numpy as np
import pandas as pd
from Layer import Layer
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
        # Fungsi aktivasi per layer
        functions = self.convertInt(re.split(' ',numbers[2]))

        n_neurons = 0
        for n in neurons[:-1]:
            n_neurons += n

        neuronArr = []
        layers = []
        weights = []

        # Bobot Layer
        counter = 1
        index = 0
        for i in range (3, n_neurons + 3):
            # Bobot per node diassign ke neuron
            weightNeuron = self.convertFloat(re.split(' ',numbers[i]))
            # Buat neuron baru
            newNeuron = Neuron(weightNeuron)
            neuronArr.append(newNeuron)

            # Bobot untuk Layer
            weights.append(weightNeuron)

            if (counter == neurons[index]):
                weights = self.transposeMatrix(weights)
                # Buat layer baru
                isHiddenLayer = False
                if index > 0 :
                    isHiddenLayer = True
                newLayer = Layer(index,neuronArr,functions[index], weights, isHiddenLayer)
                # Append layer ke Classifier
                layers.append(newLayer)
            
                print(index)
                print(weights)

                # Empty weights arr
                weights = []
                # Empty neuron arr
                neuronArr = []
                # Counter jumlah neuron per layer di ulang
                counter = 1
                # Index layer
                index += 1
            else :
                counter += 1

        # Backward Link Weight
        index = 1
        counter = 1
        weights = []
        for i in range(n_neurons + 3, len(numbers)):
            weight = self.convertFloat(re.split(' ',numbers[i]))
            weights.append(weight)

            if counter == len(layers[index].getHTValue()) :
                layers[index].setBackwardLinkWeight(weights)
                
                print(index)
                print(weights)
                print(layers[index].getHTValue())

                hTValue = []
                counter = 1
                index += 1
            else: 
                counter += 1

        # Layer output
        # Neuron di layer output
        for i in range(neurons[-1]):
            newNeuron = Neuron([])
            neuronArr.append(newNeuron)
        # Layer pada output
        newLayer = Layer(index, neuronArr, functions[-1], [])
        layers.append(newLayer)
        self.layers = layers

    def feedFoward(self):
        print("Forward")

LSTM = LSTMClassifier()

LSTM.load("text.txt")
LSTM.loadCSV('bitcoin_price_training.csv', 'bitcoin_price_test.csv')
LSTM.feedFoward()