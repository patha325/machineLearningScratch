import inputData as inData
import trainer as trainer
import neuralNetwork as network
import neuralNetwork2 as network2
import numpy as np


#Train network with inputData

net1 = False
net2 = True

if net1:
	data = inData.Input()
	NN = network.NeuralNetwork(Lambda=0.0001)
	T = trainer.Trainer(NN)
	T.train(data.trainX,data.trainY,data.testX,data.testY)

elif net2:
	data = inData.Input()
	NN = network2.NeuralNetwork2()
	#yhat = NN.forward(data.trainX)
	T = trainer.Trainer(NN)
	T.train(data.trainX,data.trainY,data.testX,data.testY)
	#print(yhat)
	#print(data.trainY)

else:
	li = [1,2,3]
	a = 5
	li = [a] + li
	print(li)
