"""Check analytical backpropagation against central finite differences."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from part1 import MLP2, PROJECT_ROOT, np, softmax


def main():
    results = {}
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 2))
    for mode in ('regression', 'classification'):
        model = MLP2(2, 3, 2, mode=mode, seed=8)
        y = rng.normal(size=(5, 2)) if mode == 'regression' else np.array([0, 1, 1, 0, 1])
        model.forward(X)
        analytical = model.backward(y)
        max_error = 0.0
        for name, gradient in zip(('W1', 'b1', 'W2', 'b2'), analytical):
            parameter = getattr(model, name)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                model.forward(X)
                plus = model.loss(y)
                parameter[index] = original - 1e-6
                model.forward(X)
                minus = model.loss(y)
                parameter[index] = original
                numeric = (plus - minus) / 2e-6
                max_error = max(max_error, abs(numeric - gradient[index]))
                np.testing.assert_allclose(numeric, gradient[index], atol=1e-7, rtol=1e-5)
        results[mode + '_gradient_max_absolute_error'] = float(max_error)
    probs = softmax(np.array([[10000., 10001., -10000.]]))
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    results['stable_softmax'] = 'passed'
    out = PROJECT_ROOT / 'results' / 'part1'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
