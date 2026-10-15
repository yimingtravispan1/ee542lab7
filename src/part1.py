"""Lab 7 Part I: NumPy MLP and the three examples from lab7.pdf.

Run from the project root: python src/part1.py
Network calculations use only NumPy; Matplotlib is used for saved figures.
"""
from pathlib import Path
import sys

# Use project-local dependencies when installed with --target .packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_local_packages = PROJECT_ROOT / ".packages"
if _local_packages.is_dir():
    sys.path.insert(0, str(_local_packages))

import math
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def d_relu(x):
    return (x > 0).astype(x.dtype)

def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)  # stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y

class MLP2:
    """
    2-layer MLP:
      hidden = ReLU(X @ W1 + b1)
      out    = hidden @ W2 + b2
    mode: 'regression' (MSE) or 'classification' (softmax CE)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, mode='regression', seed=0):
        rng = np.random.default_rng(seed)
        k1 = math.sqrt(2.0/in_dim)       # He init for ReLU
        k2 = math.sqrt(2.0/hidden_dim)
        self.W1 = rng.normal(0, k1, (in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, k2, (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))
        assert mode in ('regression','classification')
        self.mode = mode

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2

        if self.mode == 'classification':
            self.probs = softmax(self.z2)
            return self.probs
        return self.z2  # regression raw output

    def loss(self, y):
        if self.mode == 'regression':
            # y: shape (N, out_dim)
            diff = self.z2 - y
            return 0.5 * np.mean(np.sum(diff*diff, axis=1))
        else:
            # y: integer labels shape (N,)
            N = y.shape[0]
            # cross-entropy
            logp = -np.log(self.probs[np.arange(N), y] + 1e-12)
            return np.mean(logp)

    def backward(self, y):
        N = self.X.shape[0]
        if self.mode == 'regression':
            # dL/dz2 = (z2 - y)
            dz2 = (self.z2 - y) / N
        else:
            # dL/dz2 = probs - onehot(y)
            Y = one_hot(y, self.z2.shape[1])
            dz2 = (self.probs - Y) / N

        # Gradients for layer 2
        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        # Backprop through ReLU
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * d_relu(self.z1)

        # Gradients for layer 1
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, grads, lr=1e-2, weight_decay=0.0):
        dW1, db1, dW2, db2 = grads

        # L2 regularization on weights (not biases)
        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def batch_iter(X, y, batch_size, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])

    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]


def train(model, X, y, epochs=200, lr=1e-2, batch_size=64, weight_decay=0.0, verbose=True):
    losses = []
    for e in range(1, epochs+1):
        for Xb, yb in batch_iter(X, y, batch_size, shuffle=True, seed=e):
            model.forward(Xb)
            grads = model.backward(yb)
            model.step(grads, lr=lr, weight_decay=weight_decay)
        # track loss full-batch
        model.forward(X)
        L = model.loss(y)
        losses.append(L)
        if verbose and (e % max(1, epochs//10) == 0):
            print(f"epoch {e:4d}/{epochs}  loss={L:.4f}")
    return losses


def target(x):
    return np.maximum(0, 0.5*x + 0.2) + 0.3*np.maximum(0, -x + 0.5)


def rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    theta1 = rng.uniform(0, 2*np.pi, n2)
    r1 = inner_r + noise*rng.normal(size=n2)
    x1 = np.c_[r1*np.cos(theta1), r1*np.sin(theta1)]
    theta2 = rng.uniform(0, 2*np.pi, n-n2)
    r2 = inner_r + gap + noise*rng.normal(size=n-n2)
    x2 = np.c_[r2*np.cos(theta2), r2*np.sin(theta2)]
    return np.vstack([x1, x2]), np.array([0]*n2 + [1]*(n-n2))


def three_blobs(n=900, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[0, 0], [2.5, 0.5], [-2.0, 1.5]])
    cov = np.array([[0.4, 0.0], [0.0, 0.4]])
    Xs, ys = [], []
    for k, m in enumerate(means):
        Xs.append(rng.multivariate_normal(m, cov, size=n//3))
        ys.append(np.full(n//3, k))
    return np.vstack(Xs), np.hstack(ys)


def main():
    import contextlib
    import csv
    import io
    import json
    import platform
    import os
    from time import perf_counter

    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = PROJECT_ROOT / 'results' / 'part1'
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'python': platform.python_version(), 'numpy': np.__version__}
    histories = {}
    log = io.StringIO()

    def fit(name, model, X, y, **kwargs):
        model.forward(X)
        initial = float(model.loss(y))
        start = perf_counter()
        with contextlib.redirect_stdout(log):
            print('\n' + name)
            losses = train(model, X, y, **kwargs)
        duration = perf_counter() - start
        assert np.isfinite(losses).all(), name + ': nonfinite loss'
        assert losses[-1] < initial, name + ': loss did not decrease'
        histories[name] = losses
        metrics[name] = dict(initial_loss=initial, final_loss=float(losses[-1]),
                             training_seconds=duration, samples=len(X), **kwargs)

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (400, 1))
    y = target(X) + 0.05*rng.normal(size=(400, 1))
    m_reg = MLP2(1, 32, 1, mode='regression', seed=1)
    fit('regression', m_reg, X, y, epochs=500, lr=1e-2,
        batch_size=64, weight_decay=1e-4)
    xt = np.linspace(-2, 2, 9).reshape(-1, 1)
    table = np.hstack([xt, target(xt), m_reg.forward(xt)])
    np.savetxt(out / 'regression_predictions.csv', table, delimiter=',',
               header='x,target,prediction', comments='')
    metrics['regression']['train_mse'] = float(np.mean((m_reg.forward(X)-y)**2))
    grid = np.linspace(-2, 2, 401).reshape(-1, 1)
    metrics['regression']['clean_grid_mse'] = float(
        np.mean((m_reg.forward(grid)-target(grid))**2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X[:, 0], y[:, 0], s=9, alpha=0.35, label='Noisy training samples')
    ax.plot(grid[:, 0], target(grid)[:, 0], label='Target', linewidth=2)
    ax.plot(grid[:, 0], m_reg.forward(grid)[:, 0], '--', label='MLP prediction')
    ax.set(xlabel='x', ylabel='y', title='1D regression: 1 -> 32 -> 1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'regression.png', dpi=160)
    plt.close(fig)

    for name, Xc, yc, test_X, test_y, classes, seed, decay in [
        ('binary', *rings(800, 0.6, 0.5, 0.07, seed=1),
         *rings(2000, 0.6, 0.5, 0.07, seed=101), 2, 2, 1e-4),
        ('multiclass', *three_blobs(900, seed=4),
         *three_blobs(3000, seed=104), 3, 3, 0.0),
    ]:
        model = MLP2(2, 64, classes, mode='classification', seed=seed)
        fit(name, model, Xc, yc, epochs=200, lr=5e-3,
            batch_size=64, weight_decay=decay)
        probs = model.forward(Xc)
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
        metrics[name]['train_accuracy'] = float((probs.argmax(1) == yc).mean())
        # Supplement the PDF's training accuracy with independently generated data.
        metrics[name]['independent_test_samples'] = len(test_X)
        metrics[name]['independent_test_accuracy'] = float(
            (model.forward(test_X).argmax(1) == test_y).mean())
        cm = np.zeros((classes, classes), dtype=int)
        np.add.at(cm, (test_y, model.forward(test_X).argmax(1)), 1)
        metrics[name]['test_confusion_matrix_rows_true_cols_predicted'] = cm.tolist()
        np.savetxt(out / (name + '_test_confusion.csv'), cm, fmt='%d', delimiter=',')

        lower, upper = Xc.min(axis=0)-0.3, Xc.max(axis=0)+0.3
        xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 220),
                             np.linspace(lower[1], upper[1], 220))
        labels = model.forward(np.c_[xx.ravel(), yy.ravel()]).argmax(1).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contourf(xx, yy, labels, levels=np.arange(classes+1)-0.5,
                    cmap='coolwarm', alpha=0.25)
        ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap='coolwarm', s=10,
                   edgecolors='none')
        ax.set(xlabel='x1', ylabel='x2', title=name + ' classification', aspect='equal')
        fig.tight_layout()
        fig.savefig(out / (name + '_boundary.png'), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    with (out / 'losses.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'epoch', 'loss'])
        for ax, (name, losses) in zip(axes, histories.items()):
            ax.plot(np.arange(1, len(losses)+1), losses)
            ax.set(xlabel='Epoch', ylabel='Data loss', title=name)
            for epoch, loss in enumerate(losses, 1):
                writer.writerow([name, epoch, loss])
    fig.tight_layout()
    fig.savefig(out / 'loss_curves.png', dpi=160)
    plt.close(fig)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    log.write('\nRegression points: x, target, prediction\n' + str(table) + '\n')
    log.write(json.dumps(metrics, indent=2) + '\n')
    (out / 'run.log').write_text(log.getvalue(), encoding='utf-8')
    print(log.getvalue())
    print('Results saved to:', out)


if __name__ == '__main__':
    main()

