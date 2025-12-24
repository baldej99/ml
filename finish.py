

import numpy as np

np.random.seed(0)

class Tensor:
    _next_id = 0

    def __init__(self, data, creators=None, op=None, autograd=False, id=None):
        self.data = np.array(data, dtype=float)
        self.creators = creators
        self.op = op
        self.autograd = autograd

        self.grad = None
        self.children = {}

        self._cache = {}

        if id is None:
            self.id = Tensor._next_id
            Tensor._next_id += 1
        else:
            self.id = id

        if creators is not None:
            for c in creators:
                c.children[self.id] = c.children.get(self.id, 0) + 1

    def __repr__(self):
        return f"Tensor({self.data})"

    def _ensure_tensor(self, x):
        return x if isinstance(x, Tensor) else Tensor(x, autograd=False)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        if self.autograd or other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        if self.autograd or other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        if self.autograd or other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        if self.autograd or other.autograd:
            return Tensor(self.data / other.data, [self, other], "/", True)
        return Tensor(self.data / other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data, [self], "neg", True)
        return Tensor(-self.data)

    def dot(self, other):
        other = self._ensure_tensor(other)
        if self.autograd or other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.T, [self], "transpose", True)
        return Tensor(self.data.T)

    def sum(self, axis=None):
        if self.autograd:
            ax = -999 if axis is None else axis
            return Tensor(self.data.sum(axis=axis), [self], f"sum:{ax}", True)
        return Tensor(self.data.sum(axis=axis))

    def mean(self):
        if self.autograd:
            return Tensor(self.data.mean(), [self], "mean", True)
        return Tensor(self.data.mean())

    def expand(self, axis, copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [copies]
        expanded = self.data.repeat(copies).reshape(expand_shape).transpose(transpose)
        if self.autograd:
            return Tensor(expanded, [self], f"expand:{axis}", True)
        return Tensor(expanded)

    def relu(self):
        out = np.maximum(0, self.data)
        if self.autograd:
            t = Tensor(out, [self], "relu", True)
            t._cache["relu_mask"] = (self.data > 0).astype(float)
            return t
        return Tensor(out)

    def tanh(self):
        out = np.tanh(self.data)
        if self.autograd:
            t = Tensor(out, [self], "tanh", True)
            t._cache["tanh_out"] = out
            return t
        return Tensor(out)

    def sigmoid(self):
        x = np.clip(self.data, -50, 50)
        out = 1.0 / (1.0 + np.exp(-x))
        if self.autograd:
            t = Tensor(out, [self], "sigmoid", True)
            t._cache["sigmoid_out"] = out
            return t
        return Tensor(out)

    def softmax(self):
        x = self.data
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        out = exp / np.sum(exp, axis=1, keepdims=True)
        if self.autograd:
            t = Tensor(out, [self], "softmax", True)
            t._cache["softmax_out"] = out
            return t
        return Tensor(out)

    def sqrt(self, eps=1e-8):
        out = np.sqrt(self.data + eps)
        if self.autograd:
            t = Tensor(out, [self], "sqrt", True)
            t._cache["sqrt_eps"] = eps
            return t
        return Tensor(out)

    def _children_done(self):
        return all(v == 0 for v in self.children.values())

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        if grad is None:
            grad_data = np.ones_like(self.data)
        else:
            grad_data = grad.data if isinstance(grad, Tensor) else np.array(grad, dtype=float)

        if grad_origin is not None:
            self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = Tensor(grad_data, autograd=False)
        else:
            self.grad.data += grad_data

        if self.creators is None:
            return

        if not (self._children_done() or grad_origin is None):
            return

        op = self.op

        # binary ops
        if op == "+":
            a, b = self.creators
            a.backward(Tensor(self.grad.data), grad_origin=self)
            b.backward(Tensor(self.grad.data), grad_origin=self)

        elif op == "-":
            a, b = self.creators
            a.backward(Tensor(self.grad.data), grad_origin=self)
            b.backward(Tensor(-self.grad.data), grad_origin=self)

        elif op == "*":
            a, b = self.creators
            a.backward(Tensor(self.grad.data * b.data), grad_origin=self)
            b.backward(Tensor(self.grad.data * a.data), grad_origin=self)

        elif op == "/":
            a, b = self.creators
            a.backward(Tensor(self.grad.data / b.data), grad_origin=self)
            b.backward(Tensor(-self.grad.data * a.data / (b.data ** 2)), grad_origin=self)

        elif op == "neg":
            (a,) = self.creators
            a.backward(Tensor(-self.grad.data), grad_origin=self)

        elif op == "dot":
            a, b = self.creators
            G = self.grad.data
            a.backward(Tensor(G.dot(b.data.T)), grad_origin=self)
            b.backward(Tensor(a.data.T.dot(G)), grad_origin=self)

        elif op == "transpose":
            (a,) = self.creators
            a.backward(Tensor(self.grad.data.T), grad_origin=self)

        elif op.startswith("sum:"):
            (a,) = self.creators
            ax = int(op.split(":")[1])
            if ax == -999:
                a.backward(Tensor(np.ones_like(a.data) * self.grad.data), grad_origin=self)
            else:
                copies = a.data.shape[ax]
                expanded = np.repeat(np.expand_dims(self.grad.data, axis=ax), copies, axis=ax)
                a.backward(Tensor(expanded), grad_origin=self)

        elif op == "mean":
            (a,) = self.creators
            n = a.data.size
            a.backward(Tensor(np.ones_like(a.data) * (self.grad.data / n)), grad_origin=self)

        elif op.startswith("expand:"):
            (a,) = self.creators
            ax = int(op.split(":")[1])
            summed = self.grad.data.sum(axis=ax)
            a.backward(Tensor(summed), grad_origin=self)

        elif op == "relu":
            (a,) = self.creators
            mask = self._cache["relu_mask"]
            a.backward(Tensor(self.grad.data * mask), grad_origin=self)

        elif op == "tanh":
            (a,) = self.creators
            t_out = self._cache["tanh_out"]
            a.backward(Tensor(self.grad.data * (1 - t_out ** 2)), grad_origin=self)

        elif op == "sigmoid":
            (a,) = self.creators
            s = self._cache["sigmoid_out"]
            a.backward(Tensor(self.grad.data * (s * (1 - s))), grad_origin=self)

        elif op == "softmax":
            (a,) = self.creators
            y = self._cache["softmax_out"]
            g = self.grad.data
            dot = np.sum(g * y, axis=1, keepdims=True)
            dx = y * (g - dot)
            a.backward(Tensor(dx), grad_origin=self)

        elif op == "sqrt":
            (a,) = self.creators
            eps = self._cache.get("sqrt_eps", 1e-8)
            dx = self.grad.data * (0.5 / np.sqrt(a.data + eps))
            a.backward(Tensor(dx), grad_origin=self)

        elif op == "softmax_ce":
            (logits,) = self.creators
            probs = self._cache["probs"]
            target = self._cache["target"]
            B = probs.shape[0]
            g = self.grad.data
            dlogits = (probs - target) * (g / B)
            logits.backward(Tensor(dlogits), grad_origin=self)

        else:
            raise RuntimeError(f"Unknown op in backward: {op}")


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self, clip_value=1.0):
        for p in self.params:
            if p.grad is None:
                continue
            np.clip(p.grad.data, -clip_value, clip_value, out=p.grad.data)
            p.data -= self.lr * p.grad.data
            p.grad = None  # reset

class Layer:
    def get_parameters(self):
        return []

class Linear(Layer):
    def __init__(self, in_features, out_features):
        # He init
        w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        b = np.zeros(out_features, dtype=float)
        self.weight = Tensor(w, autograd=True)
        self.bias = Tensor(b, autograd=True)

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weight) + self.bias.expand(0, x.data.shape[0])

    def get_parameters(self):
        return [self.weight, self.bias]

class Tanh(Layer):
    def forward(self, x): return x.tanh()

class ReLU(Layer):
    def forward(self, x): return x.relu()

class Sigmoid(Layer):
    def forward(self, x): return x.sigmoid()

class Softmax(Layer):
    def forward(self, x): return x.softmax()

class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params



class MSELoss(Layer):
    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        diff = pred - true
        return (diff * diff).mean()

class RMSELoss(Layer):
    def forward(self, pred: Tensor, true: Tensor, eps=1e-8) -> Tensor:
        diff = true - pred
        return (diff * diff).mean().sqrt(eps=eps)

class SoftmaxCrossEntropyLoss(Layer):
    def forward(self, logits: Tensor, target_onehot: Tensor, eps=1e-12) -> Tensor:
        x = logits.data
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        ce = -np.sum(target_onehot.data * np.log(probs + eps), axis=1).mean()

        out = Tensor(ce, creators=[logits], op="softmax_ce", autograd=True)
        out._cache["probs"] = probs
        out._cache["target"] = target_onehot.data
        return out


def one_hot(y, num_classes):
    y = np.asarray(y, dtype=int)
    oh = np.zeros((len(y), num_classes), dtype=float)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def accuracy(pred_probs, y_true):
    y_pred = np.argmax(pred_probs, axis=1)
    return (y_pred == y_true).mean()

def train_regression(model, loss_fn, opt, X, Y, epochs=20000, batch=64, clip=1.0, log_every=2000):
    n = X.shape[0]
    for e in range(1, epochs + 1):
        idx = np.random.randint(0, n, size=batch)
        xb = Tensor(X[idx], autograd=False)
        yb = Tensor(Y[idx], autograd=False)

        pred = model.forward(xb)
        loss = loss_fn.forward(pred, yb)
        loss.backward()
        opt.step(clip_value=clip)

        if e % log_every == 0:
            print(f"{type(loss_fn).__name__} epoch {e} loss {float(loss.data):.6f}")

def train_classification(model, loss_fn, opt, X, Y_onehot, epochs=5000, batch=64, clip=1.0, log_every=500):
    n = X.shape[0]
    for e in range(1, epochs + 1):
        idx = np.random.randint(0, n, size=batch)
        xb = Tensor(X[idx], autograd=False)
        yb = Tensor(Y_onehot[idx], autograd=False)

        logits_or_probs = model.forward(xb)
        loss = loss_fn.forward(logits_or_probs, yb)
        loss.backward()
        opt.step(clip_value=clip)

        if e % log_every == 0:
            print(f"{type(loss_fn).__name__} epoch {e} loss {float(loss.data):.6f}")

def make_mul_dataset(n=5000, low=1, high=10, holdout=(3, 5, 4)):
    X, Y = [], []
    while len(X) < n:
        a = np.random.randint(low, high + 1)
        b = np.random.randint(low, high + 1)
        c = np.random.randint(low, high + 1)
        if (a, b, c) == holdout:
            continue
        X.append([a, b, c])
        Y.append([a * b * c])
    return np.array(X, dtype=float), np.array(Y, dtype=float)

def eval_mul(model):
    x = Tensor([[3.0, 5.0, 4.0]], autograd=False)
    x.data /= 10.0
    pred = model.forward(x).data.item() * 1000.0
    return pred

def run_mul_experiment():
    X, Y = make_mul_dataset()
    X = X / 10.0
    Y = Y / 1000.0

    print("=== Train multiplication with MSE ===")
    model = Sequential([Linear(3, 64), Tanh(), Linear(64, 64), Tanh(), Linear(64, 1)])
    opt = SGD(model.get_parameters(), lr=0.01)
    train_regression(model, MSELoss(), opt, X, Y, epochs=20000, batch=64, clip=1.0)
    mse_pred = eval_mul(model)
    print("MSE pred(3,5,4):", mse_pred)

    print("\n=== Train multiplication with RMSE ===")
    model = Sequential([Linear(3, 64), Tanh(), Linear(64, 64), Tanh(), Linear(64, 1)])
    opt = SGD(model.get_parameters(), lr=0.01)
    train_regression(model, RMSELoss(), opt, X, Y, epochs=20000, batch=64, clip=1.0)
    rmse_pred = eval_mul(model)
    print("RMSE pred(3,5,4):", rmse_pred)

    print("\n--- Conclusion (printed by script) ---")
    print("Target for (3,5,4) is 60.")
    print("Compare which is closer: MSE_pred vs RMSE_pred.")

def make_gender_dataset(n=4000):
    """
    Синтетика (пример): классы сильно пересекаются, но разделимы.
    label 0 = male, 1 = female
    """
    n2 = n // 2
    # male
    age_m = np.random.normal(35, 12, size=n2).clip(18, 70)
    w_m = np.random.normal(82, 12, size=n2).clip(50, 130)
    # female
    age_f = np.random.normal(33, 12, size=n2).clip(18, 70)
    w_f = np.random.normal(67, 12, size=n2).clip(40, 120)

    X = np.vstack([
        np.stack([w_m, age_m], axis=1),
        np.stack([w_f, age_f], axis=1),
    ])
    y = np.array([0] * n2 + [1] * n2, dtype=int)

    p = np.random.permutation(len(X))
    X, y = X[p], y[p]

    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-8)  # weight
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)  # age
    return X.astype(float), y

def eval_gender_model(model, X_test, y_test, output_mode):
    xb = Tensor(X_test, autograd=False)
    out = model.forward(xb).data
    if output_mode == "sigmoid":
        probs = out
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    elif output_mode == "softmax":
        probs = out
    elif output_mode == "logits":
        x = out - np.max(out, axis=1, keepdims=True)
        e = np.exp(x)
        probs = e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError(output_mode)
    return accuracy(probs, y_test)

def run_gender_experiments():
    X, y = make_gender_dataset(n=6000)
    oh = one_hot(y, 2)

    split = int(0.8 * len(X))
    X_train, y_train, oh_train = X[:split], y[:split], oh[:split]
    X_test, y_test = X[split:], y[split:]

    hidden_sizes = [4, 8, 16, 32, 64]

    results = []

    for h in hidden_sizes:
        print(f"\n=== Gender: hidden={h}, output=Sigmoid, loss=MSE ===")
        model = Sequential([Linear(2, h), Tanh(), Linear(h, 2), Sigmoid()])
        opt = SGD(model.get_parameters(), lr=0.05)
        train_classification(model, MSELoss(), opt, X_train, oh_train, epochs=4000, batch=64, clip=1.0, log_every=800)
        acc = eval_gender_model(model, X_test, y_test, output_mode="sigmoid")
        print("Test accuracy:", acc)
        results.append(("Sigmoid+MSE", h, acc))

    for h in hidden_sizes:
        print(f"\n=== Gender: hidden={h}, output=Softmax, loss=MSE ===")
        model = Sequential([Linear(2, h), Tanh(), Linear(h, 2), Softmax()])
        opt = SGD(model.get_parameters(), lr=0.05)
        train_classification(model, MSELoss(), opt, X_train, oh_train, epochs=4000, batch=64, clip=1.0, log_every=800)
        acc = eval_gender_model(model, X_test, y_test, output_mode="softmax")
        print("Test accuracy:", acc)
        results.append(("Softmax+MSE", h, acc))

    for h in hidden_sizes:
        print(f"\n=== Gender: hidden={h}, output=None(logits), loss=SoftmaxCE ===")
        model = Sequential([Linear(2, h), Tanh(), Linear(h, 2)])  # logits
        opt = SGD(model.get_parameters(), lr=0.05)
        train_classification(model, SoftmaxCrossEntropyLoss(), opt, X_train, oh_train, epochs=3000, batch=64, clip=1.0, log_every=600)
        acc = eval_gender_model(model, X_test, y_test, output_mode="logits")
        print("Test accuracy:", acc)
        results.append(("Logits+SoftmaxCE", h, acc))

    best = max(results, key=lambda t: t[2])
    print("\n================ SUMMARY ================")
    for name, h, acc in sorted(results, key=lambda t: (-t[2], t[1], t[0])):
        print(f"{name:16s} hidden={h:3d}  acc={acc:.4f}")
    print("\nBEST:", best[0], "hidden=", best[1], "acc=", best[2])
    print("=========================================")

if __name__ == "__main__":
    print("########################################")
    print("# Task 1-2: multiplication (MSE vs RMSE)")
    print("########################################")
    run_mul_experiment()

    print("\n\n########################################")
    print("# Task 3-5: gender classification experiments")
    print("########################################")
    run_gender_experiments()
