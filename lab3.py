import numpy as np
def NeuralNetwork(inp, weights):
    return inp.dot(weights)
prediction = NeuralNetwork(np.array([150,40]), [0.2,0.3])
true_prediction = 50
def get_error(prediction, true_prediction):
    return (true_prediction - prediction) ** 2
print(get_error(prediction, true_prediction))
weights = np.array([0.2,0.3])
while get_error(NeuralNetwork(np.array([150,40]), weights), true_prediction) > 0.001:
    delta = (NeuralNetwork(np.array([150,40]), weights) - true_prediction) * np.array([150, 40])
    print(delta)
    weights -= 0.00001 * delta
#градиентный спуск
inp = 1.2
weight = 0.8
true_prediction = 1.5
def neuralNetwork(inps, weights):
    prediction = inps * weights
    return prediction

for i in range(7):
    prediction = neuralNetwork(inp, weight)
    error = get_error(true_prediction,prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * inp
    weight -= delta;


