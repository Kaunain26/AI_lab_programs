import numpy as np

# X = (hours sleeping, hours studying), y = test score of the student
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X / np.amax(X, axis=0)  # maximum of X array
y = y / 100  # maximum test score is 100


def sigmoid(s, deriv=False):
    if deriv == True:
        return s * (1 - s)
    return 1 / (1 + np.exp(-s))


class NeuralNetwork:
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = sigmoid(self.z3)
        return output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * sigmoid(self.z2, deriv=True)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.output_delta)

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)


NN = NeuralNetwork()

for i in range(1000):
    NN.train(X, y)

print("Input: " + str(X))
print("Actual Output: " + str(y))
print("Predicted Output: " + str(NN.feedForward(X)))
