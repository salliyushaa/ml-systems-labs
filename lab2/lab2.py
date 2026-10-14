import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0
y = digits.target

y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.30, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


class MLP:
    def __init__(self, in_features=64, hidden_dim=16, num_classes=10):
        self.W1 = np.random.normal(0.0, 0.01, (in_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0.0, 0.01, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.normal(0.0, 0.01, (hidden_dim, num_classes))
        self.b3 = np.zeros(num_classes)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        exps = np.exp(self.z3 - np.max(self.z3, axis=1, keepdims=True))
        self.a3 = exps / np.sum(exps, axis=1, keepdims=True)
        return self.a3

    def loss(self, pred, targets):
        return -np.sum(targets * np.log(pred + 1e-15)) / targets.shape[0]

    def backward(self, x, targets):
        batch_sz = x.shape[0]

        dz3 = (self.a3 - targets) / batch_sz
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def apply_grads(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        self.W3 -= lr * grads["W3"]
        self.b3 -= lr * grads["b3"]


test_model = MLP(hidden_dim=16)
p = test_model.forward(X_train[:32])
print("Output shape:", p.shape)
print("Sums equal 1:", np.allclose(np.sum(p, axis=1), 1.0, atol=1e-6))
print(f"Initial loss: {test_model.loss(p, y_train[:32]):.4f}")

np.random.seed(123)
chk_net = MLP(hidden_dim=16)
x_val = X_train[:8]
y_val = y_train[:8]

chk_net.forward(x_val)
g = chk_net.backward(x_val, y_val)

eps = 1e-5
passed = 0
tests = [
    ("W1", chk_net.W1, g["W1"]),
    ("W2", chk_net.W2, g["W2"]),
    ("W3", chk_net.W3, g["W3"])
]

for idx in range(10):
    name, matrix, d_matrix = tests[idx % len(tests)]
    r = np.random.randint(matrix.shape[0])
    c = np.random.randint(matrix.shape[1])

    val = matrix[r, c]

    matrix[r, c] = val + eps
    l_pos = chk_net.loss(chk_net.forward(x_val), y_val)

    matrix[r, c] = val - eps
    l_neg = chk_net.loss(chk_net.forward(x_val), y_val)

    matrix[r, c] = val

    num_val = (l_pos - l_neg) / (2.0 * eps)
    ana_val = d_matrix[r, c]

    diff = np.abs(num_val - ana_val) / max(1e-8, (np.abs(num_val) + np.abs(ana_val)))
    ok = diff < 1e-6
    if ok:
        passed += 1
    print(
        f"Check {idx + 1} [{name}]: Num={num_val:.6e}, Ana={ana_val:.6e}, RelError={diff:.2e} -> {'PASS' if ok else 'FAIL'}")

print(f"Gradient Check Result: {passed}/10 passed\n")

os.makedirs("results", exist_ok=True)
epochs = 50
bs = 32
results = {}

for hidden_w in [16, 64]:
    for rate in [0.001, 0.01, 0.1]:
        tag = f"W={hidden_w}, lr={rate}"
        net = MLP(hidden_dim=hidden_w)
        losses = []

        for ep in range(epochs):
            perm = np.random.permutation(len(X_train))
            x_shuff = X_train[perm]
            y_shuff = y_train[perm]

            cur_losses = []
            for i in range(0, len(X_train), bs):
                xb = x_shuff[i:i + bs]
                yb = y_shuff[i:i + bs]

                out = net.forward(xb)
                cur_losses.append(net.loss(out, yb))
                grads = net.backward(xb, yb)
                net.apply_grads(grads, rate)

            losses.append(np.mean(cur_losses))

        final_test_out = net.forward(X_test)
        acc = np.mean(np.argmax(final_test_out, axis=1) == np.argmax(y_test, axis=1))
        print(f"Final Test Accuracy for [{tag}]: {acc * 100:.2f}%")
        results[tag] = (losses, acc)

plt.figure(figsize=(9, 5))
for key, (hist, acc) in results.items():
    plt.plot(hist, label=f"{key} ({acc * 100:.1f}%)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curves.png")
plt.close()

prof_net = MLP(hidden_dim=64)
x_b = X_train[:32]
y_b = y_train[:32]

f_times = []
b_times = []

for _ in range(100):
    start = time.perf_counter()
    prof_net.forward(x_b)
    f_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    prof_net.backward(x_b, y_b)
    b_times.append(time.perf_counter() - start)

print(f"Forward pass median time: {np.median(f_times) * 1e6:.2f} µs")
print(f"Backward pass median time: {np.median(b_times) * 1e6:.2f} µs")
