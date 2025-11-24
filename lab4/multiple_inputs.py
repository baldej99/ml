import numpy as np
def NeuralNetwork(inp, weights):
    return inp.dot(weights)
def get_error(prediction, true_prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.00001

for i in range(1000):
    prediction = NeuralNetwork(inp, weights)
    error = get_error(prediction, true_prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta
    delta[0] = 0
    weights -= delta



