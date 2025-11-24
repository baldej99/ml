import numpy as np
def relu(x):
    return np.maximum(0, x)

inp = np.array([
[15, 10],
[15, 15],
[15, 20],
[25, 10]
])
true_prediction = np.array([[10, 20, 15, 20]])

layer_hid_size = 3
layer_in_size = len(inp[0])
layer_out_size = 1

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
print(weights_hid.shape)
weights_hid_2 = 2 * np.random.random((layer_hid_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1
print(weights_out.shape)

print(weights_hid)
print(weights_out)

prediction_hid = relu(inp[0].dot(weights_hid))
print(prediction_hid.shape)
prediction_hid_2 = relu(prediction_hid.dot(weights_hid_2))
prediction = prediction_hid_2.dot(weights_out)
print(prediction)
