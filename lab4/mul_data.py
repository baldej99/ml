import numpy as np

def neural_networks(inps, weigths):
    return inps.dot(weigths)

def RMSE(true_predictions, predictions):
    errors = (np.array(true_predictions) - np.array(predictions)) ** 2
    return np.sqrt(np.mean(errors))

inps = np.array([[150, 40],
                [170, 80],
                [160, 90]])

true_predictions = np.array([50, 120, 140])

weights = np.array([0.2, 0.3])

learning_rate = 0.00001

for i in range(500):
    predictions = []
    for j in range(len(inps)):
        inp = inps[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(inp, weights)
        predictions.append(prediction)
        print('Prediction: %.10f, Weights: %s, True prediction: %.10f' % (prediction, weights, true_prediction))
        delta = (prediction - true_prediction) * inp * learning_rate
        weights -= delta
    print("rmse: %.10f" % (RMSE(true_predictions, predictions)))
    print("------------------")






