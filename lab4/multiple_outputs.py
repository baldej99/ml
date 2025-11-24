import numpy as np
def NeuralNetwork(inp, weights):
    return inp * weights
def get_error(prediction, true_prediction):
    return (true_prediction - prediction) ** 2

inp = 150
weights = np.array([0.2, 0.3])
true_predictions = np.array([50, 120])
learning_rate = 0.00001

for i in range(100):
    prediction = NeuralNetwork(inp, weights)
    error = get_error(prediction, true_predictions)
    print("Prediction: %s, Weights: %s, Error: %s" % (prediction, weights, error))
    delta = (prediction - true_predictions) * inp * learning_rate
    weights -= delta



