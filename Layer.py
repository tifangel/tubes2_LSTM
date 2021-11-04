class Layer:
    def __init__(self, idlayer, neurons, activfunc, weights, hidden=False):
        self.id_layer = idlayer
        self.activfunc = activfunc
        self.neurons = neurons
        self.weights = weights
        self.backwardLinkWeight = []
        self.bias = [0.1 for i in range(len(weights))]
        self.output = []

        if hidden :
            self.hTValue = [0.1 for i in range(len(weights[0]))]

    def getNeurons(self):
        return self.neurons

    def getWeights(self): 
        return self.weight

    def getHTValue(self):
        return self.hTValue

    def getOutput(self):
        return self.output

    def setNeurons(self, neurons):
        self.neurons = neurons

    def setHTValue(self, hTValue):
        self.hTValue = hTValue

    def setBackwardLinkWeight(self, weights):
        self.backwardLinkWeight = weights

    def setNeuronHVal(self, index, val):
        self.neurons[index].setHValue(val)

    