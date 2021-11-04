
class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.hValue = 0
        self.target = [] # Nilai y untuk layer output
        self.sigma = 0

    def printWeight(self):
        print(self.weights)
    
    def getHValue(self):
        return self.hValue

    def getWeights(self):
        return self.weights

    def getTargetValue(self):
        return self.target

    def getSigma(self):
        return self.sigma

    def setSigma(self, sigma):
        self.sigma = sigma

    def setHValue(self, h):
        self.hValue = h

    def setWeights(self, w):
        self.weights = w
    
    def setTargetValue(self, t):
        self.target = t