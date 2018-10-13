#!/usr/bin/python3
import numpy as np
import time
from mnist_loader import Dataset
import sys
'''
Element wise multiplication involves with * product
and matrix multiplication involves with numpy's dot
'''


class Network:

	def __init__(self, layers=None):
		# prepare dataset
		self.dataset = Dataset(
						mini_batch_size=16,
						length=60000,
						test_length=10000
					)
		'''Weights are row matrices and biases are coloumn vectors
			i.e. in case of a having 10 neurons each connected to 784 
			neurons, the weight matrix would be of a 10x784 order 2-D
			matrix. The bias would be 10x1 orders column vector.'''
		self.layersNo= len(layers)
		self.layers  = layers
		self.biases  = [np.random.randn(y, 1) for y in layers[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(layers[:-1], layers[1:])]
		self.nabla_b = [np.zeros(b.shape) for b in self.biases]
		self.nabla_w = [np.zeros(w.shape) for w in self.weights]
		self.a       = None
		self.z       = None
		self.cost    = None
		self.avgCost = None

	@staticmethod
	def sigmoid(z):
		return 1/(1+np.exp(-z))

	@staticmethod
	def sigmoid_derivative(a):
		return a*(1-a)

	def forwardprop(self, a):
		# inputs are the activatiin for input layer
		Z, A = [None], [a]
		for w,b in zip(self.weights, self.biases):
			z = np.dot(w, a)+b
			a = self.sigmoid(z)
			A.append(a), Z.append(z)
		self.z, self.a = Z, A

	def think(self, a):
		for w,b in zip(self.weights, self.biases):
			a = self.sigmoid(np.dot(w, a)+b)
		return a

	def cost_function(self, y):
		return np.sum(np.power(np.subtract(self.a[-1],y),2))/(2.0*self.dataset.mini_batch_size)

	def nabla_functions(self, y, lr=0.1):
		cost_derivative = np.subtract(self.a[-1], y)
		delta = cost_derivative*self.sigmoid_derivative(self.a[-1])
		self.nabla_w[-1] = np.dot(delta, self.a[-2].T)
		self.nabla_b[-1] = delta
		for i in range(2, self.layersNo):
			delta = np.dot(self.weights[-i+1].transpose(), delta) * self.sigmoid_derivative(self.a[-i])
			self.nabla_b[-i] = delta
			self.nabla_w[-i] = np.dot(delta, self.a[-i-1].transpose())

	def backwardprop(self, input_data, input_label):
		# CALCULATE COST FUNCTION
		self.cost = self.cost_function(input_label)
		# nabla_w and nabla_b
		self.nabla_functions(input_label)
		# update weights and biases with nabla values
		self.update_weights_biases()

	def update_weights_biases(self, lr=0.01):
		self.weights = [w-(lr/self.dataset.mini_batch_size)*nw 
						for w, nw in zip(self.weights, self.nabla_w)]
		self.biases = [b-(lr/self.dataset.mini_batch_size)*nb 
						for b, nb in zip(self.biases, self.nabla_b)]

	# Training
	def train(self, epochs=None):
		epochs = 100 if epochs == None else epochs
		mb = 0 
		training_started = time.time()
		print("Training is in progress...")
		for epoch in range(epochs):    
			mini_batches_d, mini_batches_l = self.dataset.get_mini_batches
			print("[+] Epoch %s is in progress..."%(epoch))
			for mini_batch_d, mini_batch_l in zip(mini_batches_d, mini_batches_l):
				for input_data, input_label in zip(mini_batch_d, mini_batch_l):
					input_data = np.array(np.matrix(input_data).T)
					self.forwardprop(input_data)
					self.backwardprop(input_data, input_label)
					self.update_weights_biases()
				mb += 1
			print("[+] Epoch %s:"%(epoch),end=' ')
			self.evaluate()
		# print out relevant info.
		print("Training completed in", str(round(time.time() - training_started, 2)) + 's.')

	def prediction(self, out):
		i = 0
		m = np.max(out)
		for o in out:
			if m==o[0]:
				break
			i += 1
		return i

	def evaluate(self):
		i = 1
		correct = 0
		for data, label in zip(self.dataset.test_data, self.dataset.test_label):
			td = np.array(np.matrix(data).T)
			r = self.prediction(label)
			o = self.prediction(self.think(td))
			status = 1 if r==o else 0
			correct += status
			"""print("Test %s:Output %s, Predicted %s, Status %s, Accuracy %s"%(
					i, r, o, status, round(correct/float(i), 2)
				)
			)"""
			i+=1
		print("accuracy is " + str(100*(correct/float(i))) +" %")


if __name__ == "__main__":
	if len(sys.argv) > 1:
		epochs = int(sys.argv[1])
	else:
		epochs = None
	nn = Network([784, 16, 16, 10])
	nn.train(epochs=epochs)
	#nn.evaluate()
