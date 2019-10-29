import inputData as inData
import trainer as trainer
import neuralNetwork as network


#Train network with inputData

data = inData.Input()
NN = network.NeuralNetwork(Lambda=0.0001)

T = trainer.Trainer(NN)
T.train(data.trainX,data.trainY,data.testX,data.testY)