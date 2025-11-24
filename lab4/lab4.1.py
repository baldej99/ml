import numpy as np
def neuralNetwork(inps, weights):
    prediction = inps * weights
    return prediction
inp = 30
weight = 0.2
true_prediction = 70
learning_rate = 0.001

def get_error(prediction, true_prediction):
    return (true_prediction - prediction) ** 2

for i in range(100):
    prediction = neuralNetwork(inp, weight)
    error = get_error(true_prediction,prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weight -= delta;


