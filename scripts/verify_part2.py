"""Check NumPy CNN operations against the PDF loops and numerical gradients."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from part2 import PROJECT_ROOT, CatDogCNN, conv2d, maxpool, mse_loss, np


def main():
    rng = np.random.default_rng(123)
    images = rng.normal(size=(3, 8, 9, 3))
    kernel = rng.normal(size=(3, 3, 3))
    expected = np.zeros((3, 6, 7))
    for n in range(3):
        for i in range(6):
            for j in range(7):
                expected[n, i, j] = np.sum(images[n, i:i+3, j:j+3] * kernel)
    np.testing.assert_allclose(conv2d(images, kernel), expected, atol=1e-12)
    np.testing.assert_allclose(conv2d(images[0], kernel), expected[0], atol=1e-12)
    pooled = np.array([[[np.max(image[i:i+2, j:j+2]) for j in range(0, 6, 2)]
                        for i in range(0, 6, 2)] for image in expected])
    np.testing.assert_allclose(maxpool(expected), pooled)
    np.testing.assert_allclose(maxpool(expected[0]), pooled[0])

    model = CatDogCNN(seed=0)
    output, flat = model.forward(rng.uniform(0, 1, (2, 64, 64, 3)))
    assert flat.shape == (2, 961) and output.shape == (2, 1)
    model.W_fc.fill(0.001)
    model.b_fc.fill(0.5)  # Away from the ReLU kink for finite differences.
    features = flat[0]
    label = np.array([1.0])
    gradient = 2 * (model.predict_features(features) - label)
    analytic_w = np.outer(features, gradient)
    analytic_b = gradient.copy()
    maximum_error = 0.0
    for parameter, analytic in ((model.W_fc, analytic_w), (model.b_fc, analytic_b)):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = mse_loss(label, model.predict_features(features))
            parameter[index] = original - 1e-6
            minus = mse_loss(label, model.predict_features(features))
            parameter[index] = original
            numeric = (plus - minus) / 2e-6
            maximum_error = max(maximum_error, abs(numeric - analytic[index]))
            np.testing.assert_allclose(numeric, analytic[index], atol=1e-8, rtol=1e-5)
    fixed_filter = model.conv_filter.copy()
    before = mse_loss(label, model.predict_features(features))
    model.step(features, label)
    assert mse_loss(label, model.predict_features(features)) < before
    np.testing.assert_array_equal(model.conv_filter, fixed_filter)

    # Negative output pre-activations must have zero ReLU gradient.
    model.W_fc.fill(0)
    model.b_fc.fill(-1)
    model.step(features, label)
    assert (model.W_fc == 0).all() and model.b_fc[0] == -1
    results = {'convolution_matches_pdf_loops': True,
               'pooling_matches_pdf_loops_including_odd_width': True,
               'flattened_features': 961, 'fc_gradient_max_absolute_error': maximum_error,
               'sgd_reduces_loss_on_fixture': True, 'convolution_remains_fixed': True,
               'inactive_output_relu_has_zero_gradient': True}
    out = PROJECT_ROOT / 'results/part2'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
