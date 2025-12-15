import numpy as np
from keras.src.datasets import mnist

def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return x > 0

train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = x_train[:train_images_count].reshape(train_images_count, pixels_per_image) / 255
train_labels = y_train[:train_images_count]

test_images = x_test[:test_images_count].reshape(test_images_count, pixels_per_image) / 255
test_labels = y_test[:test_images_count]

one_hot_labels = np.zeros((len(train_labels), digits_num))
for i, label in enumerate(train_labels):
    one_hot_labels[i, label] = 1
train_labels = one_hot_labels

one_hot_labels = np.zeros((len(test_labels), digits_num))
for i, label in enumerate(test_labels):
    one_hot_labels[i, label] = 1
test_labels = one_hot_labels

np.random.seed(123)
hidden_size = 100
weight_hid = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weight_out = 0.2 * np.random.random((hidden_size, digits_num)) - 0.1

learning_rate = 0.1
num_epochs = 100
batch_size = 100

for i in range(num_epochs):
    correct_predictions = 0
    for j in range(int(train_images.shape[0] / batch_size)):
        batch_start = j * batch_size
        batch_end = (j + 1) * batch_size
        layer_in = train_images[batch_start:batch_end]
        layer_hid = relu(layer_in.dot(weight_hid))
        dropout_mask = np.random.binomial(n=1, p=0.35, size=layer_hid.shape)
        layer_hid *= dropout_mask
        layer_out = layer_hid.dot(weight_out)
        correct_predictions += np.sum(np.argmax(layer_out, axis=1) == np.argmax(train_labels[batch_start:batch_end], axis=1))
        layer_out_delta = (layer_out - train_labels[batch_start:batch_end])/batch_size
        layer_hid_delta = layer_out_delta.dot(weight_out.T) * relu_deriv(layer_hid) * dropout_mask
        weight_out -= layer_hid.T.dot(layer_out_delta) * learning_rate
        weight_hid -= layer_in.T.dot(layer_hid_delta) * learning_rate
    print("Epoch: ", i + 1)
    print("Accuracy: %.2f" % (correct_predictions * 100 / len(train_images)))

correct_answers = 0
for j in range(len(test_images)):
    layer_in = test_images[j:j+1]
    layer_hid = relu(np.dot(layer_in, weight_hid))
    layer_out = np.dot(layer_hid, weight_out)
    correct_answers += int(np.argmax(layer_out) == np.argmax(test_labels[j:j + 1]))
print("Accuracy: %.2f" %(correct_answers * 100/len(test_images)))