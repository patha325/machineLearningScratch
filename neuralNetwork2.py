import numpy as np

class NeuralNetwork2(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 2  #Number of neurons for the layer.
        self.outputLayerSize = 1 #Number of neurons for the layer.
        self.hiddenLayerSize = 8 #3 #Number of neurons for the layer.
        self.hiddenLayerDecrease = 0 #1 #Number of neurons removed for each depth of hidden layer.
        self.hiddenLayerDepth = 15#3
        self.Lambda = Lambda
        
        #Weights (parameters)
        self.w = [np.random.randn(self.inputLayerSize,self.hiddenLayerSize)] # inputLayer

        for i in range(0,self.hiddenLayerDepth):
            self.w.append(np.random.randn(self.hiddenLayerSize-i*self.hiddenLayerDecrease,self.hiddenLayerSize-(i+1)*self.hiddenLayerDecrease)) # hiddenLayers

        self.w.append(np.random.randn(self.hiddenLayerSize-self.hiddenLayerDepth*self.hiddenLayerDecrease,self.outputLayerSize)) # outputLayer

    def forward(self, X):
        #Propogate inputs though network
        self.z = [np.dot(X, self.w[0])]
        self.a = [self.sigmoid(self.z[0])]

        for i in range(0,self.hiddenLayerDepth+1):
            self.z.append(np.dot(self.a[i], self.w[1+i]))
            self.a.append(self.sigmoid(self.z[1+i]))

        yHat = self.a[-1]        
        return yHat

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0]
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        self.delta = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[-1]))
        self.dJdW = []

        for i in range(0,self.hiddenLayerDepth+1):
            self.dJdW = [np.dot(self.a[len(self.a)-2-i].T, self.delta)/X.shape[0]] + self.dJdW
            self.delta = np.dot(self.delta, self.w[len(self.w)-1-i].T)*self.sigmoidPrime(self.z[len(self.z)-2-i])

        self.dJdW = [np.dot(X.T, self.delta)/X.shape[0]] + self.dJdW

        return self.dJdW

    def getParams(self):
        #Get W Rolled into a vector:
        returnVal = []
        first = True
        # hidden layers + 2
        for i in range(0,self.hiddenLayerDepth+2):
            if first:
                returnVal = self.w[i].ravel()
                first = False
            else:
                returnVal = np.concatenate((returnVal,self.w[i].ravel()))

        return returnVal
    
    def setParams(self, params):
        #Set W using single parameter vector:
        start = 0
        w_end = 0
        #print("params",params)
        for i in range(self.hiddenLayerDepth+1):
            w_end += self.w[i].shape[0]*self.w[i].shape[1]
            #print(i,params[start:w_end])
            self.w[i] = np.reshape(params[start:w_end], \
                             (self.w[i].shape[0], self.w[i].shape[1]))
            start = w_end

    def computeGradients(self, X, y):

        returnVal = []
        # hidden layers + 2
        dJdW = self.costFunctionPrime(X, y)
        for i in range(self.hiddenLayerDepth+2):
            returnVal = np.concatenate((returnVal,dJdW[i].ravel()))

        return returnVal

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)