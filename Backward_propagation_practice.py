import numpy as np

x = np.array(([2, 9], [1, 9], [3, 8]), dtype=float)
y = np.array(([90], [86], [89]), dtype=float)

x = x / np.amax(x, axis=0)
y = y / 100


def sigmoid(s, deriv=False):
    if deriv:
        return s * (1 - s)
    else:
        return 1 / (1 + np.exp(-s))


class NeuralNetwork:
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)

    def feedForward(self, x):
        self.z = np.dot(x, self.w1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        output = sigmoid(self.z3)
        return output

    def backward(self, x, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * sigmoid(self.z2, deriv=True)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.output_delta)

    def train(self, x, y):
        output = self.feedForward(x)
        self.backward(x, y, output)


NN = NeuralNetwork()

for i in range(1000):
    NN.train(x, y)

print("Input: ", str(x))
print("Actual Output: ", str(y))
print("Predicted output: ", NN.feedForward(x))
