import numpy as np

X = np.array(([2,9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # maximum of X array longitudinally
y = y / 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # (1 / 1+e^-x)


def derivatives_sigmoid(x):
    return x * (1 - x)


epoch = 1000
lr = 0.1
inputlayer_neurns = 2

hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(size=(inputlayer_neurns, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))  # bh - bias
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    # Forward Propogation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)

    # how much hidden layer wts contributed to error
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

# dotproduct of nextlayererror and currentlayerop
wout += hlayer_act.T.dot(d_output) * lr
wh += X.T.dot(d_hiddenlayer) * lr

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)

#   output............
# Input:
# [[0.2        1.        ]
#  [1.         0.55555556]
#  [0.4        0.66666667]]
# Actual Output:
# [[0.92]
#  [0.86]
#  [0.89]]
# Predicted Output:
#  [[0.80958431]
#  [0.81118552]
#  [0.8039376 ]]
