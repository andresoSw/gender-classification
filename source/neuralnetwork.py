import numpy as np

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
	return x*(1.0-x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - x*x

class NeuralNetwork:

	def __init__(self,learning_rate=0,n_in=0,n_hidden=0,n_out=0, activation='sigmoid', momentum=0):
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_derivative = sigmoid_derivative
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_derivative = tanh_derivative
		self.learning_rate = learning_rate
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.n_out = n_out
		self.momentum = momentum

	def initialize_weights(self):
		# Weights from input unit to hidden unit
		self.W1 = (0.1)*np.random.random((self.n_in+1,self.n_hidden))-0.5
		self.W1_delta = np.zeros((self.n_in+1,self.n_hidden))
		# Weights from hidden unit to output unit
		self.W2 = (0.1)*np.random.random((self.n_hidden+1,self.n_out))-0.5
		self.W2_delta = np.zeros((self.n_hidden+1,self.n_out))


	def feed_forward(self,X):
		# Add bias to input unit
		input_bias = np.ones((len(X),1))
		self.input_units = np.column_stack((X,input_bias))

		# output of input layer fed with input values
		self.a1 = self.activation(np.dot(self.input_units,self.W1))

		# Add bias to output unit
		output_bias = np.ones((len(X),1))
		self.output_units = np.column_stack((self.a1,output_bias))

		# output of hidden layer fed with output of input layer
		self.y = self.activation(np.dot(self.output_units,self.W2))
		return self.y

	# backpropagation updating weights with a single example
	def backpropagate(self,X,T):
		for i in range(0,len(X)):
			x = np.atleast_2d(X[i])
			t = np.atleast_2d(T[i])
			self.feed_forward(x)

			output_error = t - self.y
			output_delta = output_error*self.activation_derivative(self.y)

			# error in hidden layer
			hidden_error = output_delta.dot(self.W2.T)
			hidden_delta = hidden_error*self.activation_derivative(self.output_units)

			self.W2_delta = self.learning_rate*self.output_units.T.dot(output_delta) + self.momentum*self.W2_delta 
			self.W2 = self.W2 + self.W2_delta

			self.W1_delta = self.learning_rate*self.input_units.T.dot(hidden_delta)[...,:-1] + self.momentum*self.W1_delta 
			self.W1 = self.W1 + self.W1_delta


	# backpropagation updating weights with all examples
	def backpropagate_batch(self,X,T):
		self.feed_forward(X)
		# error in output layer
		output_error = T - self.y
		output_delta = output_error*self.activation_derivative(self.y)

		# error in hidden layer
		hidden_error = output_delta.dot(self.W2.T)
		hidden_delta = hidden_error*self.activation_derivative(self.output_units)

		self.W2_delta = self.learning_rate*self.output_units.T.dot(output_delta) + self.momentum*self.W2_delta 
		self.W2 = self.W2 + self.W2_delta

		self.W1_delta = self.learning_rate*self.input_units.T.dot(hidden_delta)[...,:-1] + self.momentum*self.W1_delta 
		self.W1 = self.W1 + self.W1_delta

	# backpropagation for a number of N iteration
	def backpropagation(self,X,T,maxIterations,batch = False,file_name=None):
		if batch: 
			for i in range(0,maxIterations):
				self.backpropagate_batch(X,T)
				estimation = self.feed_forward(X)
				estimationError = EstimationError(estimatedValues=estimation,targetValues=T)
				estimationError.computeErrors()
				totalError = estimationError.getTotalError()
				print i,' ',totalError/len(X)
				if file_name is not None:
					results_file = ''.join([file_name.rsplit('.', 1)[0]]+['.out'])
					with open(results_file,'a') as results_data:
						results_data.write(str(i))
						results_data.write(',')
						results_data.write((str(totalError/len(X))))
						results_data.write('\n')
		else:
			for i in range(0,maxIterations):
				self.backpropagate(X,T)
				estimation = self.feed_forward(X)
				estimationError = EstimationError(estimatedValues=estimation,targetValues=T)
				estimationError.computeErrors()
				totalError = estimationError.getTotalError()
				print i,' ',totalError/len(X)
				if file_name is not None:
					results_file = ''.join([file_name.rsplit('.', 1)[0]]+['.out'])
					with open(results_file,'a') as results_data:
						results_data.write(str(i))
						results_data.write(',')
						results_data.write((str(totalError/len(X))))
						results_data.write('\n')


class EstimationError:

	def __init__(self,estimatedValues,targetValues,errors=[]):
		assert (isinstance(estimatedValues,np.ndarray) and isinstance(targetValues,np.ndarray)),"expected numpy ndarray as input,got %s and %s instead" %(type(estimatedValues),type(targetValues))

		self.estimatedValues = estimatedValues
		self.targetValues = targetValues
		self.errors = errors

	def computeErrors(self):
		self.errors = np.power((self.targetValues - self.estimatedValues),2)

	def getTotalError(self):
		return sum(self.errors)
