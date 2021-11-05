import numpy as np

class Layer:
    def __init__(self, idlayer, n_unit, n_unit_next, hidden=False):
        self.id_layer = idlayer
        self.neurons = [0 for i in range(n_unit)]
        self.wf = np.random.random_sample((n_unit_next,n_unit))
        self.wi = np.random.random_sample((n_unit_next,n_unit))
        self.wc = np.random.random_sample((n_unit_next,n_unit))
        self.wo = np.random.random_sample((n_unit_next,n_unit))
        self.bias = np.random.random_sample((n_unit_next,))
        self.output = [0 for i in range(n_unit_next)]
        if hidden :
            self.bwf = np.random.random_sample((n_unit,n_unit))
            self.bwi = np.random.random_sample((n_unit,n_unit))
            self.bwc = np.random.random_sample((n_unit,n_unit))
            self.bwo = np.random.random_sample((n_unit,n_unit))
            self.hTValue = [0 for i in range(n_unit)]
            self.cTValue = [0 for i in range(n_unit)]

    def getNeurons(self):
        return self.neurons

    def getWeightsForgetGate(self): 
        return self.wf
    
    def getWeightsInputGate(self): 
        return self.wi
    
    def getWeightsCellGate(self): 
        return self.wc
    
    def getWeightsOutputGate(self): 
        return self.wo

    def getBackwardWeightsForgetGate(self): 
        return self.bwf
    
    def getBackwardWeightsInputGate(self): 
        return self.bwi
    
    def getBackwardWeightsCellGate(self): 
        return self.bwc
    
    def getBackwardWeightsOutputGate(self): 
        return self.bwo
    
    def getBias(self): 
        return self.bias

    def getHTValue(self):
        return self.hTValue
    
    def getCTValue(self):
        return self.cTValue

    def getOutput(self):
        return self.output

    def setNeurons(self, neurons):
        self.neurons = neurons

    def setOutput(self, output):
        self.output = output

    def setHTValue(self, hTValue):
        self.hTValue = hTValue

    def setCTValue(self, cTValue):
        self.cTValue = cTValue
    