import numpy as np

class Tensor(object):
    _next_id = 0

    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.autograd = autograd
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.children = {}

        if id is None:
            self.id = Tensor._next_id
            Tensor._next_id += 1
        else:
            self.id = id

        if creators is not None:
            for creator in creators:
                creator.children[self.id] = creator.children.get(self.id, 0) + 1


    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        if self.autograd or other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd or other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd or other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data, [self], "-1", True)
        return Tensor(-self.data)

    def dot(self, other):
        if self.autograd or other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.T, [self], "transpose", True)
        return Tensor(self.data.T)

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))

        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        expand_data = expand_data.transpose(transpose)

        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    def relu(self):
        out = np.maximum(0, self.data)
        if self.autograd:
            return Tensor(out, [self], "relu", True)
        return Tensor(out)

    def tanh(self):
        out = np.tanh(self.data)
        if self.autograd:
            return Tensor(out, [self], "tanh", True)
        return Tensor(out)

    def sigmoid(self):
        x = np.clip(self.data, -50, 50)
        out = 1.0 / (1.0 + np.exp(-x))
        if self.autograd:
            return Tensor(out, [self], "sigmoid", True)
        return Tensor(out)

    def check_grads_from_children(self):
        return all(v == 0 for v in self.children.values())

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        if grad is None:
            grad_data = np.ones_like(self.data)
        else:
            grad_data = grad.data if isinstance(grad, Tensor) else np.array(grad)

        if grad_origin is not None:
            self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = Tensor(grad_data, autograd=False)
        else:
            self.grad.data += grad_data

        if self.creators is None:
            return
        if not (self.check_grads_from_children() or grad_origin is None):
            return

        op = self.operation_on_creation

        if op == "+":
            self.creators[0].backward(Tensor(self.grad.data), grad_origin=self)
            self.creators[1].backward(Tensor(self.grad.data), grad_origin=self)

        elif op == "-1":
            self.creators[0].backward(Tensor(-self.grad.data), grad_origin=self)

        elif op == "-":
            self.creators[0].backward(Tensor(self.grad.data), grad_origin=self)
            self.creators[1].backward(Tensor(-self.grad.data), grad_origin=self)

        elif op == "*":
            a = self.creators[0].data
            b = self.creators[1].data
            self.creators[0].backward(Tensor(self.grad.data * b), grad_origin=self)
            self.creators[1].backward(Tensor(self.grad.data * a), grad_origin=self)

        elif op.startswith("sum_"):
            axis = int(op.split("_")[1])
            copies = self.creators[0].data.shape[axis]
            expanded = np.repeat(np.expand_dims(self.grad.data, axis=axis), copies, axis=axis)
            self.creators[0].backward(Tensor(expanded), grad_origin=self)

        elif op.startswith("expand_"):
            axis = int(op.split("_")[1])
            summed = self.grad.data.sum(axis=axis)
            self.creators[0].backward(Tensor(summed), grad_origin=self)

        elif op == "transpose":
            self.creators[0].backward(Tensor(self.grad.data.T), grad_origin=self)

        elif op == "dot":
            A = self.creators[0].data
            B = self.creators[1].data
            G = self.grad.data
            self.creators[0].backward(Tensor(G.dot(B.T)), grad_origin=self)   # dL/dA
            self.creators[1].backward(Tensor(A.T.dot(G)), grad_origin=self)   # dL/dB

        elif op == "relu":
            x = self.creators[0].data
            mask = (x > 0).astype(float)
            self.creators[0].backward(Tensor(self.grad.data * mask), grad_origin=self)
        elif op == "tanh":
            x = self.creators[0].data
            grad = 1 - np.tanh(x) ** 2
            self.creators[0].backward(
                Tensor(self.grad.data * grad),
                grad_origin=self
            )
        elif op == "sigmoid":
            x = np.clip(self.creators[0].data, -50, 50)
            s = 1.0 / (1.0 + np.exp(-x))
            self.creators[0].backward(
                Tensor(self.grad.data * (s * (1.0 - s))),
                grad_origin=self)


class SGD(object):
    def __init__(self, weights, learning_rate=0.01):
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        for w in self.weights:
            if w.grad is None:
                continue
            w.data -= self.learning_rate * w.grad.data
            w.grad.data *= 0



np.random.seed(546154)

def make_dataset(n=3000, low=1, high=10):
    X, Y = [], []
    while len(X) < n:
        a = np.random.randint(low, high + 1)
        b = np.random.randint(low, high + 1)
        c = np.random.randint(low, high + 1)
        if (a, b, c) == (3, 5, 4):   # do not train on the test triple
            continue
        X.append([a, b, c])
        Y.append([a * b * c])
    return np.array(X, dtype=float), np.array(Y, dtype=float)

X_train, y_train = make_dataset()

# normalize (important)
X_train /= 10.0
y_train /= 1000.0

# weights (NO bias)
W1 = Tensor(np.random.randn(3, 16) * 0.2, autograd=True)
W2 = Tensor(np.random.randn(16, 16) * 0.2, autograd=True)
W3 = Tensor(np.random.randn(16, 1) * 0.2, autograd=True)

weights = [W1, W2, W3]
sgd = SGD(weights, learning_rate=0.0005)

epochs = 30000
batch_size = 128

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], size=batch_size)
    xb = Tensor(X_train[idx], autograd=False)
    yb = Tensor(y_train[idx], autograd=False)

    # forward (use tanh, works much better than relu without bias)
    h = xb.dot(W1).tanh()
    h = h.dot(W2).tanh()
    pred = h.dot(W3)

    diff = pred - yb
    loss = (diff * diff).sum(0).sum(0)   # scalar

    loss.backward()
    sgd.step()

    if (epoch + 1) % 1000 == 0:
        print("epoch", epoch + 1, "loss", float(loss.data))

x_test = Tensor([[3.0, 5.0, 4.0]], autograd=False)
x_test.data /= 10.0

h = x_test.dot(W1).tanh()
h = h.dot(W2).tanh()
pred = h.dot(W3)

pred_value = pred.data.item() * 1000.0
print("Prediction for (3,5,4):", pred_value)

