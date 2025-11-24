import numpy as np

def neural_networks(inps, weigths):
    return inps.dot(weigths)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([70, 90])
weights = np.array([[0.2, 0.3], [0.5, 0.7]]).T
true_prediction = np.array([50,120])
learning_rate = 0.00001

for i in range(10):
    predictions = neural_networks(inp, weights)
    error = get_error(true_prediction, predictions)
    print("Weights: %s, Error: %s" % (weights, error))
    delta = (predictions - true_prediction) * inp * learning_rate
    weights = weights - delta



