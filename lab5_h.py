import numpy as np

def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - x ** 2


x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))
print(weight_hid.shape)

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    #sigmoid

    #layer_hid = sigmoid(x.dot(weight_hid))
    #layer_out = sigmoid(layer_hid.dot(weight_out))

    #relu

    #layer_hid = relu(x.dot(weight_hid))
    #layer_out = relu(layer_hid.dot(weight_out))

    #tan_h

    layer_hid = tanh(np.dot(x, weight_hid))
    layer_out = tanh(layer_hid.dot(weight_out))

    error = (layer_out - y) ** 2
    #sigmoid

    #layer_out_delta = (layer_out - y) * sigmoid_deriv(layer_out)
    #layer_hid_delta = layer_out_delta.dot(weight_out.T) * sigmoid_deriv(layer_hid)

    #relu

    #layer_out_delta = (layer_out - y) * relu_deriv(layer_out)
    #layer_hid_delta = layer_out_delta.dot(weight_out.T) * relu_deriv(layer_hid)

    #tanh

    layer_out_delta = (layer_out - y) * tanh_deriv(layer_out)
    layer_hid_delta = layer_out_delta.dot(weight_out.T) * tanh_deriv(layer_hid)

    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hid_delta)

    if (epoch % 100 == 0):
        error = np.mean(error)
        print(f"Epoch: {epoch}, Error: {error}")
input = np.array([[0, 1]])
#sigmoid

#layer_hid = sigmoid(input.dot(weight_hid))
#layer_out = sigmoid(layer_hid.dot(weight_out))

#relu

#layer_hid = relu(input.dot(weight_hid))
#layer_out = relu(layer_hid.dot(weight_out))

#tanh

layer_hid = tanh(input.dot(weight_hid))
layer_out = tanh(layer_hid.dot(weight_out))
print("prediction ", layer_out)