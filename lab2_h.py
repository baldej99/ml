import numpy as np
def relu_der(x):
    return x > 0

def relu(x):
    return (x > 0) * x

learning_rate = 0.00001
num_epochs = 3000

inp = np.array([
[15, 10],
[15, 15],
[15, 20],
[25, 10]
])
true_prediction = np.array([10, 1569, 15, 20])

layer_hid_size = 400
layer_in_size = len(inp[0])
layer_out_size = 1

np.random.seed(123)

weight_hid =  2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weight_out = np.random.random((layer_hid_size, layer_out_size))

for i in range(num_epochs):
    out_error = 0
    for i in range(len(inp)):
        layer_in = inp[i:i+1]
        layer_hid = relu(layer_in.dot(weight_hid))
        layer_out = layer_hid.dot(weight_out)
        out_error += np.sum(layer_out - true_prediction[i:i+1])**2
        out_delta = true_prediction[i:i+1] - layer_out
        hid_delta = out_delta.dot(weight_out.T)*relu_der(layer_hid)
        weight_out += learning_rate * layer_hid.T.dot(out_delta)
        weight_hid += learning_rate * layer_in.T.dot(hid_delta)
        print("Predictions: %s, true_predictions: %s" % (layer_out, true_prediction[i:i + 1]))
        print("Errors: %.4f" % out_error)
    print("----------------------")


