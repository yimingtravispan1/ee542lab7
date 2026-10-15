"""Part II: fixed NumPy convolution + ReLU + pooling, trained FC + ReLU.

Prepare data: python scripts/prepare_part2_data.py
Run: python src/part2.py
"""
from pathlib import Path
import argparse
import csv
import json
import os
import platform
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / '.packages'))
import numpy as np
from PIL import Image


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def conv2d(images, kernel):
    """Valid stride-one RGB cross-correlation, as in the lab (no kernel flip)."""
    single = images.ndim == 3
    batch = images[None] if single else images
    kh, kw, channels = kernel.shape
    if batch.ndim != 4 or batch.shape[-1] != channels:
        raise ValueError('Expected NHWC images and an HWC kernel')
    windows = np.lib.stride_tricks.sliding_window_view(batch, (kh, kw), axis=(1, 2))
    result = np.einsum('bhwcij,ijc->bhw', windows, kernel, optimize=False)
    return result[0] if single else result


def maxpool(images, size=2):
    single = images.ndim == 2
    batch = images[None] if single else images
    n, h, w = batch.shape
    oh, ow = h // size, w // size
    cropped = batch[:, :oh * size, :ow * size]
    pooled = cropped.reshape(n, oh, size, ow, size).max(axis=(2, 4))
    return pooled[0] if single else pooled


class CatDogCNN:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv_filter = rng.standard_normal((3, 3, 3)) * 0.1
        # 64 -> valid 3x3 convolution: 62 -> 2x2 pooling: 31 -> 961.
        self.W_fc = rng.standard_normal((31 * 31, 1)) * 0.01
        self.b_fc = np.zeros(1)

    def features(self, images):
        return maxpool(relu(conv2d(images, self.conv_filter))).reshape(len(images), -1)

    def predict_features(self, features):
        return relu(features @ self.W_fc + self.b_fc)

    def forward(self, images):
        flat = self.features(images)
        return self.predict_features(flat), flat

    def step(self, flat, label, lr=0.001):
        predicted = self.predict_features(flat)
        gradient = mse_grad(label, predicted) * relu_derivative(predicted)
        self.W_fc -= lr * np.outer(flat, gradient)
        self.b_fc -= lr * gradient.reshape(self.b_fc.shape)
        return mse_loss(label, predicted)


def load_images(data, records):
    images = np.empty((len(records), 64, 64, 3), dtype=np.float64)
    for index, record in enumerate(records):
        with Image.open(data / record['path']) as image:
            # Pillow's default RGB resize filter is BICUBIC; make it explicit.
            rgb = image.convert('RGB').resize((64, 64), Image.Resampling.BICUBIC)
            images[index] = np.asarray(rgb, dtype=np.float64) / 255.0
    return images


def extract_features(model, data, records, batch_size):
    features = np.empty((len(records), 31 * 31), dtype=np.float64)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        features[start:start + len(batch)] = model.features(load_images(data, batch))
    return features


def load_manifest(data):
    path = data / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError('Prepare images first: python scripts/prepare_part2_data.py')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    train, test = [], []
    seen = set()
    for record in manifest['images']:
        if record['sha256'] in seen:
            raise ValueError('Duplicate image in split manifest')
        seen.add(record['sha256'])
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid image path')
        if not (data / relative).is_file():
            raise FileNotFoundError(data / relative)
        if relative.parts[0] not in ('train', 'test'):
            raise ValueError('Unknown split')
        if relative.parts[1] != ('cat', 'dog')[record['label']]:
            raise ValueError('Folder and label disagree')
        (train if relative.parts[0] == 'train' else test).append(record)
    for split in (train, test):
        if {r['label'] for r in split} != {0, 1}:
            raise ValueError('Both classes must be present in each split')
    return manifest, train, test


def run(args):
    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    start_run = perf_counter()
    data = PROJECT_ROOT / 'data'
    out = PROJECT_ROOT / 'results' / 'part2'
    out.mkdir(parents=True, exist_ok=True)
    manifest, training, testing = load_manifest(data)
    rng = np.random.default_rng(args.seed)
    training = [training[i] for i in rng.permutation(len(training))]
    testing = [testing[i] for i in rng.permutation(len(testing))]
    model = CatDogCNN(args.seed)
    fixed_filter = model.conv_filter.copy()
    start = perf_counter()
    features = extract_features(model, data, training, args.feature_batch_size)
    feature_seconds = perf_counter() - start
    labels = np.array([r['label'] for r in training]).reshape(-1, 1)
    initial_loss = mse_loss(labels, model.predict_features(features))
    print(f'Training features: {features.shape}; extracted in {feature_seconds:.3f}s', flush=True)
    history, log = [], []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        total_loss = 0.0
        # Match the PDF's online SGD: one image per parameter update, fixed order.
        for flat, label in zip(features, labels):
            total_loss += model.step(flat, label, args.lr)
        full_prediction = model.predict_features(features)
        row = {'epoch': epoch, 'online_mse': total_loss / len(training),
               'post_epoch_mse': mse_loss(labels, full_prediction),
               'train_accuracy': float(((full_prediction > 0.5) == labels).mean()),
               'seconds': perf_counter() - epoch_start}
        history.append(row)
        message = f"Epoch {epoch}/{args.epochs} | Loss: {row['online_mse']:.6f} | Train accuracy: {row['train_accuracy']:.4%}"
        log.append(message)
        print(message, flush=True)
    training_seconds = perf_counter() - start
    assert np.array_equal(model.conv_filter, fixed_filter), 'Convolution filter changed'
    assert np.isfinite(model.W_fc).all() and np.isfinite(model.b_fc).all()

    blocks, prediction_rows = [], []
    confusion = np.zeros((2, 2), dtype=int)
    test_loss_sum, zero_outputs = 0.0, 0
    # Process held-out images after training; no test features enter SGD.
    for block_number, offset in enumerate(range(0, len(testing), args.block_size), 1):
        block = testing[offset:offset + args.block_size]
        block_start = perf_counter()
        flat = extract_features(model, data, block, args.feature_batch_size)
        output = model.predict_features(flat).ravel()
        predicted = (output > 0.5).astype(int)
        truth = np.array([r['label'] for r in block])
        seconds = perf_counter() - block_start
        assert np.isfinite(output).all()
        np.add.at(confusion, (truth, predicted), 1)
        test_loss_sum += float(np.sum((output - truth) ** 2))
        zero_outputs += int(np.sum(output == 0))
        row = {'block': block_number, 'samples': len(block),
               'cat_samples': int(np.sum(truth == 0)), 'dog_samples': int(np.sum(truth == 1)),
               'correct': int(np.sum(predicted == truth)),
               'accuracy': float(np.mean(predicted == truth)), 'processing_seconds': seconds,
               'images_per_second': len(block) / seconds}
        blocks.append(row)
        for record, score, prediction in zip(block, output, predicted):
            prediction_rows.append({'path': record['path'], 'true_label': record['label'],
                                    'score': float(score), 'predicted_label': int(prediction),
                                    'block': block_number})
        message = f"Test block {block_number}: {len(block)} images | accuracy {row['accuracy']:.4%} | {seconds:.3f}s"
        log.append(message)
        print(message, flush=True)
    test_seconds = sum(row['processing_seconds'] for row in blocks)
    metrics = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'platform': platform.platform(), 'processor': platform.processor(), 'device': 'CPU',
        'dataset_revision': manifest['revision'], 'split_seed': manifest['split_seed'],
        'counts': manifest['counts'], 'model_seed': args.seed,
        'architecture': '64x64x3 -> fixed valid 3x3x3 conv -> ReLU -> 2x2 maxpool -> 961 -> FC(1) -> ReLU',
        'trainable_parameters': 962, 'fixed_convolution_parameters': 27,
        'epochs': args.epochs, 'learning_rate': args.lr, 'sgd_batch_size': 1,
        'feature_batch_size': args.feature_batch_size, 'test_block_size': args.block_size,
        'dtype': 'float64', 'resize': 'Pillow RGB BICUBIC', 'prediction_threshold': 0.5,
        'training_order': 'seeded shuffle once, same order each epoch',
        'fixed_feature_reuse': True, 'initial_train_mse': initial_loss,
        'final_train_mse': history[-1]['post_epoch_mse'],
        'train_accuracy': history[-1]['train_accuracy'],
        'test_mse': test_loss_sum / len(testing),
        'test_accuracy': float(np.trace(confusion) / len(testing)),
        'test_majority_baseline': max(manifest['counts']['test'].values()) / len(testing),
        'test_confusion_rows_true_cols_predicted': confusion.tolist(),
        'test_zero_output_fraction': zero_outputs / len(testing),
        'convolution_unchanged': True,
        'timing_seconds': {'training_load_resize_conv_pool': feature_seconds,
                           'training_sgd_and_epoch_metrics': training_seconds,
                           'training_total': feature_seconds + training_seconds,
                           'test_load_resize_conv_pool_fc': test_seconds},
        'test_images_per_second': len(testing) / test_seconds,
        'test_ms_per_image': 1000 * test_seconds / len(testing),
    }
    for name, rows in [('losses.csv', history), ('test_blocks.csv', blocks),
                       ('test_predictions.csv', prediction_rows)]:
        with (out / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savetxt(out / 'test_confusion.csv', confusion, delimiter=',', fmt='%d')
    np.savez(out / 'model.npz', conv_filter=model.conv_filter, W_fc=model.W_fc, b_fc=model.b_fc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['epoch'] for r in history], [r['online_mse'] for r in history], label='Online MSE')
    axes[0].plot([r['epoch'] for r in history], [r['post_epoch_mse'] for r in history], label='Post-epoch MSE')
    axes[0].set(xlabel='Epoch', ylabel='MSE', title='Training loss')
    axes[0].legend()
    axes[1].bar([r['block'] for r in blocks], [r['accuracy'] for r in blocks])
    axes[1].set(xlabel='Unseen-image block', ylabel='Accuracy', ylim=(0, 1), title='Held-out block accuracy')
    fig.tight_layout()
    fig.savefig(out / 'training_and_evaluation.png', dpi=160)
    plt.close(fig)
    metrics['timing_seconds']['run_including_exports'] = perf_counter() - start_run
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (out / 'run.log').write_text('\n'.join(log) + '\n' + json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--feature-batch-size', type=int, default=64)
    args = parser.parse_args()
    if min(args.epochs, args.block_size, args.feature_batch_size) < 1 or args.lr <= 0:
        parser.error('epochs, batch sizes, and learning rate must be positive')
    try:
        run(args)
    except FileNotFoundError as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
