# Laboratory 7 - Part II: NumPy Cat/Dog CNN

This report addresses Part II on PDF pages 10-12. The required workflow is to extract cat/dog images from Parquet, train a small NumPy CNN with only its fully connected layer updated, evaluate large blocks of unseen images, measure image processing time, and report the results. Part III's PyTorch/GPU implementation is outside this report.

## 1. Dataset and Held-Out Images

The source is the PDF's [microsoft/cats_vs_dogs dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs/tree/b5ae3589204019bc2cc97e99e4914a54589333ef), pinned to revision `b5ae3589204019bc2cc97e99e4914a54589333ef`. Its label mapping is cat = 0 and dog = 1. Both Parquet shards were downloaded and verified against their SHA-256 hashes. All 23,410 source rows were read. No additional images failed decoding; 24 byte-identical duplicate rows were removed before splitting, leaving 23,386 unique image byte sequences.

The split is stratified by class with 80% for training and the remainder for testing, using `np.random.default_rng(42)`. Integer rounding is applied separately to each class. The full unique dataset is used, rather than a small demonstration subset.

| Split | Cat (0) | Dog (1) | Total |
| --- | ---: | ---: | ---: |
| Training | 9,380 | 9,328 | 18,708 |
| Held-out test | 2,346 | 2,332 | 4,678 |

The directory layout is `data/train/cat/`, `data/train/dog/`, `data/test/cat/`, and `data/test/dog/`. The test subfolders retain labels for automatic evaluation. `data/manifest.json` records each source shard/row, label, byte hash, and destination path. No image byte hash appears twice across the two splits. This prevents exact-file leakage; it does not establish the absence of all visually similar images. No test image is used in SGD or model selection.

Preparation is implemented in [prepare_part2_data.py](../scripts/prepare_part2_data.py). Download checksums, counts, duplicate rows, and preparation time are recorded in [data_preparation.json](../results/part2/data_preparation.json). Raw Parquet files, original image bytes, and the detailed local manifest are excluded from Git using `data/` in `.gitignore`.

## 2. Corrections to the PDF Example

The experiment preserves the PDF's one random RGB convolution filter, ReLU activations, max pooling, scalar output, MSE loss, learning rate 0.001, 10 epochs, and training of the FC layer only. Two defects in the sample code require explicit corrections:

1. **FC input dimension.** A valid stride-one 3x3 RGB filter maps a 64x64x3 input to a 62x62 scalar feature map. The filter sums over the three input channels; it does not produce three output channels. A 2x2 stride-two max pool then produces 31x31, so the flattened size is 961. The PDF's `32 * 32 * 3` FC input size would cause a matrix-dimension error. This implementation uses `W_fc.shape == (961, 1)`.
2. **Class-ordered split.** The PDF's loader appends all cats before all dogs, then slices at 80% without shuffling. On this approximately balanced dataset the resulting test set would contain only dogs. The implementation instead makes the stratified held-out split described above. Training samples are shuffled once with model seed 0 and then processed in the same order each epoch, retaining the PDF's sequential per-image updates.

Byte-level deduplication before splitting is an additional safeguard. Vectorized convolution and pooling are mathematically equivalent to the PDF loops. Since the convolution filter never changes, the implementation computes training features once and reuses them for all 10 epochs. This reduces repeated computation while preserving the per-image FC updates. Test features are computed only after training. These changes do not add trainable convolution layers or switch to a different loss/model.

## 3. Implementation and Training

The implementation is [src/part2.py](../src/part2.py). Neural-network computations and updates use NumPy; Pillow performs decoding/resizing, PyArrow reads Parquet, and Matplotlib creates the result plot. No neural-network framework or GPU is used.

| Stage | Operation | Output per image |
| --- | --- | --- |
| Preprocessing | Convert to RGB, resize to 64x64 using Pillow BICUBIC, divide by 255 | 64x64x3 |
| Convolution | One fixed 3x3x3 filter, stride 1, no padding | 62x62 |
| Hidden activation | ReLU | 62x62 |
| Pooling | Maximum over 2x2 regions, stride 2 | 31x31 |
| Flatten | Reshape | 961 |
| FC and output | Affine transform followed by ReLU | 1 |

Parameters use `np.random.default_rng(0)`: convolution entries are sampled from a standard normal distribution times 0.1; FC weights use standard normal times 0.01; FC bias starts at zero. There are 27 fixed convolution parameters and 962 trainable FC parameters (961 weights plus one bias). Calculations use float64. Hyperparameters and seed were fixed before observing held-out results.

For one image with flattened feature vector `f`, label `y`, pre-activation `z = f @ W_fc + b_fc`, and prediction `p = max(0, z)`:

```text
loss = (p - y)**2
g = 2 * (p - y) * (p > 0)
grad_W = outer(f, g)
grad_b = g
W_fc = W_fc - 0.001 * grad_W
b_fc = b_fc - 0.001 * grad_b
```

This is online SGD with one sample per update. The convolution filter receives no gradient update, as explicitly required by the simplified Part II example. An assertion confirms that it remains exactly unchanged. The predicted class is dog if the scalar output is greater than 0.5, otherwise cat. The ReLU scalar is a regression score and is not a calibrated probability.

Training records both the mean online loss before each sample update (the PDF's printed loss) and the full training MSE after each epoch. These are different measurements because parameters change during the epoch.

## 4. Measured Results and Timing

All numbers below come from the full local run on CPU, using Python 3.13.15 and NumPy 2.5.4 on Windows 11. Machine-readable outputs are [metrics.json](../results/part2/metrics.json), [losses.csv](../results/part2/losses.csv), [test_blocks.csv](../results/part2/test_blocks.csv), [test_predictions.csv](../results/part2/test_predictions.csv), and [run.log](../results/part2/run.log).

| Metric | Recorded value |
| --- | ---: |
| Initial full training MSE | 0.497860 |
| Final online training MSE (epoch 10) | 0.245325 |
| Final full training MSE | 0.244767 |
| Training accuracy | 60.4875% |
| Test MSE | 0.245212 |
| Test accuracy | 58.9996% (2,760 / 4,678) |
| Test majority-class baseline | 50.1496% |
| Training image loading, resizing, convolution, pooling | 166.4118 s |
| FC SGD and training metrics, all 10 epochs | 3.0306 s |
| Total training processing | 169.4424 s |
| Test image processing, all 10 blocks | 39.4744 s |
| Test throughput | 118.5072 images/s |
| Mean test processing time per image | 8.4383 ms |
| Whole run including exports | 209.5963 s |
| Dataset download, validation, and preparation (separate) | 80.0767 s |

The confusion matrix uses true classes as rows and predicted classes as columns:

| True / Predicted | Cat (0) | Dog (1) |
| --- | ---: | ---: |
| Cat (0) | 1,315 | 1,031 |
| Dog (1) | 887 | 1,445 |

Every held-out image is evaluated exactly once. Block accuracies and wall-clock processing times are:

| Block | Images | Correct | Accuracy | Processing time (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 512 | 297 | 58.0078% | 4.3097 |
| 2 | 512 | 304 | 59.3750% | 4.2407 |
| 3 | 512 | 308 | 60.1562% | 4.5574 |
| 4 | 512 | 297 | 58.0078% | 4.2756 |
| 5 | 512 | 311 | 60.7422% | 4.1946 |
| 6 | 512 | 305 | 59.5703% | 4.3030 |
| 7 | 512 | 294 | 57.4219% | 4.2533 |
| 8 | 512 | 305 | 59.5703% | 4.1899 |
| 9 | 512 | 296 | 57.8125% | 4.4904 |
| 10 | 70 | 43 | 61.4286% | 0.6598 |

![Training loss and held-out block accuracy](../results/part2/training_and_evaluation.png)

The final test accuracy exceeds the majority-class baseline by about 8.85 percentage points, but remains modest. The training/test accuracy gap is about 1.49 percentage points. These results suggest limited predictive features in this simplified architecture; this single split does not establish performance across all cat/dog images. Training loss decreases over the 10 epochs, while threshold-based training accuracy peaks earlier and then changes slightly. MSE and accuracy optimize different quantities, so their trends need not be identical. No checkpoint was selected using test results: the reported model is the specified epoch-10 model.

None of the test outputs is exactly zero, so the inactive-output ReLU failure mode did not collapse this particular final model. It remains a limitation of the architecture. Errors occur in both classes, as shown by the confusion matrix. No minimum accuracy is specified in PDF Part II; these are the actual baseline results rather than an accuracy guarantee.

## 5. Answers to the Part II Evaluation Tasks

**Automatically train with a set of images, then test large blocks of unseen images.** Running `python src/part2.py` trains on all 18,708 training images for 10 epochs, then evaluates all 4,678 held-out images. The default test block size is 512, producing nine full blocks and a final block of 70. Test order is deterministically shuffled using the same seeded generator after the training-order permutation. Each block includes loading, RGB conversion, resizing, normalization, convolution, ReLU, pooling, FC inference, and thresholding. Predictions are exported with their true labels, block number, and paths. Only the final block can be smaller than the requested size.

**Measure the time to process the images.** `perf_counter()` measures training image loading/preprocessing/feature extraction, FC training, and each test block separately. Test throughput is the total number of test images divided by the sum of block processing durations. Milliseconds per image use that same total. Prediction CSV assembly and file/plot exports are excluded from the block timings. Block timings include cold computation of test features; they do not report FC-only inference as whole-image processing. They are observed wall-clock timings, not a controlled cold-disk benchmark. Download/preparation is reported separately. Training feature reuse is disclosed so its timing is not mistaken for repeatedly convolving every image in every epoch.

**Submit all relevant results in the report.** Section 4 presents overall metrics, the confusion matrix, per-block accuracy/timing, and the training/evaluation plot, supported by the raw files. [model.npz](../results/part2/model.npz) stores the convolution filter, trained FC weights, and bias. Numerical verification is recorded in [verification.json](../results/part2/verification.json).

**What does this CNN learn, and what are its limitations?** Only the FC layer learns how to combine random convolution/pooling features. It cannot learn task-specific visual filters. MSE treats class labels as scalar targets, and ReLU at the output can become inactive: if its pre-activation is nonpositive, its derivative is zero and that example cannot update the FC layer. Scores are unbounded above, unlike sigmoid or Softmax probabilities. A high-level cat/dog distinction is difficult for this small fixed-feature model, so completion of the lab workflow should not be interpreted as evidence of strong recognition accuracy. More filters, training convolution weights, and a suitable classification loss could improve the model, but would change this Part II baseline.

## 6. Verification and Reproduction

[scripts/verify_part2.py](../scripts/verify_part2.py) checks batched and single-image convolution against the PDF's nested loops, pooling against its loop including an odd-width crop, the 961-element flattened shape, every FC gradient against central finite differences, a loss-reducing SGD step, unchanged convolution weights, and zero gradients for an inactive output ReLU. These are extra correctness checks, beyond the required evaluation.

All checks passed; the maximum absolute FC gradient error was `5.70e-11`. The full experiment additionally verified that the convolution remained unchanged and the learned parameters and test outputs were finite. The saved predictions reproduce the reported accuracy and confusion matrix.

Run from the project root:

```powershell
python -m pip install --target .packages -r requirements.txt
python scripts/prepare_part2_data.py
python scripts/verify_part2.py
python src/part2.py
```

Preparation reuses an existing complete manifest with the same seed/revision. It requires internet access on the first run and downloads about 722 MB. Training and verification then run offline using the local images. `--block-size` changes the evaluation block size; defaults are 512 for evaluation blocks and 64 for internal feature-extraction batches. These internal batches do not change SGD's one-image updates. All paths are resolved relative to the project root, and rerunning training replaces the Part II result files. Timings vary by machine and load.

## 7. Completion Status

All Part II tasks on PDF pages 10-12 have been implemented and executed on the full prepared dataset: Parquet extraction, labeled folders, preprocessing, the corrected NumPy CNN, FC-only training, automatic large-block held-out evaluation, image-processing timings, and this report. There are no remaining Part II data, dependency, or execution blockers. The model's approximately 59% test accuracy is a recorded performance limitation. Part III's GPU comparison and video log are separate tasks.
