import numpy as np

class Input():
	def __init__(self):
		#Training Data:
		self.trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
		self.trainY = np.array(([75], [82], [93], [70]), dtype=float)

		#Testing Data:
		self.testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
		self.testY = np.array(([70], [89], [85], [75]), dtype=float)

		#Normalize:
		self.trainX = self.trainX/np.amax(self.trainX, axis=0)
		self.trainY = self.trainY/100 #Max test score is 100

		#Normalize by max of training data:
		self.testX = self.testX/np.amax(self.trainX, axis=0)
		self.testY = self.testY/100 #Max test score is 100