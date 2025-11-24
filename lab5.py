import numpy as np

def neural_networks(inps, weigths):
    return inps.dot(weigths)

def RMSE(true_predictions, predictions):
    errors = (np.array(true_predictions) - np.array(predictions)) ** 2
    return np.sqrt(np.mean(errors))

#Объявим несколько наборов данных на два входа
inps = np.array([
[150, 40],
[140, 35],
[155, 45],
[185, 95],
[145, 40],
[195, 100],
[180, 95],
[170, 80],
[160, 90],
])
weights = np.array([0.2,0.3])
true_predictions = np.array([0,0,0,100,0,100,100,100,100])
learning_rate = 0.0001

for i in range(100):
    delta = 0
    predictions = []
    for j in range(len(inps)):
        inp = inps[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(inp, weights)
        predictions.append(prediction)
        print('Prediction: %.10f, Weights: %s, True prediction: %.10f' % (prediction, weights, true_prediction))
        delta += (prediction - true_prediction) * inp * learning_rate
    weights -= delta / len(inps)
    print("rmse: %.10f" % (RMSE(true_predictions, predictions)))
    print("------------------")

print(neural_networks(np.array([150,45]), weights))
print(neural_networks(np.array([170,85]), weights))
#5
inps2 = np.array([[10, 5],
                  [0, -5],
                  [2,6]])
true_preds2 = np.array([15, -5, 8])
np.random.seed(123)
weights2 = np.random.random(2)
for i in range(4000):
    delta = 0
    predictions = []
    for j in range(len(inps2)):
        inp = inps2[j]
        true_prediction = true_preds2[j]
        prediction = neural_networks(inp, weights2)
        predictions.append(prediction)
        print('Prediction: %.10f, Weights: %s, True prediction: %.10f' % (prediction, weights2, true_prediction))
        delta += (prediction - true_prediction) * inp * learning_rate
    weights2 -= delta / len(inps2)
    print("rmse: %.10f" % (RMSE(true_preds2, predictions)))
    print("------------------")

print(neural_networks(np.array([12, 4]), weights2))
print(neural_networks(np.array([-1000, 5000]), weights2))





