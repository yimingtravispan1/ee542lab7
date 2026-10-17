# Laboratory 7 Report 

Part I has no separate numbered question list. Sections 1-4 answer its explanatory points and discuss all three examples; Section 6 maps these responses to the PDF.

## 1. Objectives and Model

A perceptron applies an activation to a weighted input sum plus bias. An MLP connects these units in fully connected, feedforward layers. ReLU introduces nonlinearity; without it, stacked affine layers would remain a single affine mapping. More layers or neurons increase representational capacity and training cost.

The implementation is in [src/part1.py](src/part1.py). Following the PDF example code, the model has one hidden layer and two fully connected layers with trainable parameters:

```text
Input X -> Z1 = XW1 + b1 -> H1 = ReLU(Z1) -> Z2 = H1W2 + b2
Regression: return Z2 directly
Classification: return softmax(Z2)
```

The conceptual example on page 2 is `2 -> 4 -> 4 -> 2`, whereas the subsequent `MLP2` code and all three experiments have one hidden layer. This implementation follows the concrete code. "Two-layer" counts the two parameterized layers and excludes the input layer.

- `ReLU(z) = max(0, z)`. Its derivative is 1 for `z > 0` and 0 otherwise; the derivative at zero is taken as 0.
- Weights use He initialization with variance `2 / fan_in` to help preserve signal scale through ReLU; biases start at zero.
- Softmax subtracts each row's maximum before exponentiation to avoid overflow, then normalizes the outputs into probabilities.
- One-hot encoding represents each class label as a vector with 1 at the corresponding class index. Classification selects the largest Softmax probability using `argmax`.
- The regression loss follows the PDF: `L = (1 / (2*N)) * sum_i sum_j (Z2[i,j] - y[i,j])**2`. With one output, this is half the ordinary MSE.
- The classification loss is mean cross-entropy: `L = -(1/N) * sum_i log(p[i,y[i]] + 1e-12)`.

Regression returns continuous raw outputs and penalizes squared error. Classification returns class probabilities; cross-entropy penalizes low probability assigned to the true class.

## 2. Backpropagation and Training

Backpropagation applies the chain rule from the output through the second linear layer, ReLU, and the first linear layer. The output gradient is `dZ2 = (Z2 - y) / N` for regression and `dZ2 = (p - one_hot(y)) / N` for Softmax/cross-entropy. As in the PDF, the latter uses the standard gradient, ignoring the small `1e-12` logarithm safeguard. The remaining gradients are:

```text
dW2 = H1.T @ dZ2               db2 = sum(dZ2, axis=0, keepdims=True)
dH1 = dZ2 @ W2.T
dZ1 = dH1 * (Z1 > 0)
dW1 = X.T @ dZ1                db1 = sum(dZ1, axis=0, keepdims=True)
```

Here `@` denotes matrix multiplication, `.T` denotes transpose, and `*` denotes elementwise multiplication. Intermediate values `X`, `Z1`, and `H1` are cached during the forward pass because the backward pass needs them to apply the chain rule.

Each epoch shuffles samples using `seed=epoch`, then performs forward propagation, backpropagation, and SGD updates for each mini-batch. Mini-batches reduce per-update memory requirements and introduce gradient noise that can help optimization. The learning rate controls step size; loss need not decrease at every update. Weight decay discourages large weights using `W = W - lr * (dW + weight_decay * W)`; biases use `b = b - lr * db`. Full training-set data loss is measured after each epoch, excluding the regularization term.

## 3. Experimental Setup

| Experiment | Network | Training samples | Epochs | Learning rate | Batch size | Weight decay | Data/model seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1D regression | 1 -> 32 -> 1 | 400 | 500 | 0.01 | 64 | 0.0001 | 0 / 1 |
| Concentric-ring binary classification | 2 -> 64 -> 2 | 800 | 200 | 0.005 | 64 | 0.0001 | 1 / 2 |
| Three-blob multiclass classification | 2 -> 64 -> 3 | 900 | 200 | 0.005 | 64 | 0 | 4 / 3 |

Regression inputs are sampled uniformly from `[-2, 2]`. The target is `f(x) = max(0, 0.5*x + 0.2) + 0.3*max(0, -x + 0.5)`, with additive Gaussian noise of standard deviation 0.05. The concentric rings have radii 0.6 and 1.1, with radial noise of standard deviation 0.07. The three Gaussian blobs have means `(0, 0)`, `(2.5, 0.5)`, and `(-2, 1.5)`, each with covariance `diag(0.4, 0.4)`.

These parameters, training data seeds, and model seeds match the PDF. An additional 2,000 ring samples (seed=101) and 3,000 blob samples (seed=104) are generated with different seeds and used only for evaluation after training. These independent synthetic test sets follow the same distributions as the training sets.

## 4. Recorded Results

| Experiment | Initial loss | Final loss | Training accuracy |
| --- | ---: | ---: | ---: |
| 1D regression | 1.076999 | 0.001775 | N/A |
| Binary classification | 0.958286 | 0.281615 | 99.875% |
| Multiclass classification | 1.778096 | 0.144010 | 95.111% |

The recorded environment is Python 3.13.15 and NumPy 2.5.4 on CPU. As optional evaluation beyond Part I, independent ring and blob test accuracies are 99.900% and 95.233%, respectively. Detailed metrics and timings remain in [metrics.json](results/part1/metrics.json).

![Loss curves for all three experiments](results/part1/loss_curves.png)

### 4.1 1D Regression

The ordinary training MSE is **0.00354938**. The MSE against the noiseless target on a uniform grid of 401 points is **0.00099166**.

| x | Noiseless target | Prediction |
| ---: | ---: | ---: |
| -2.0 | 0.750000 | 0.732648 |
| -1.5 | 0.600000 | 0.602820 |
| -1.0 | 0.450000 | 0.472992 |
| -0.5 | 0.300000 | 0.346200 |
| 0.0 | 0.350000 | 0.259591 |
| 0.5 | 0.450000 | 0.490820 |
| 1.0 | 0.700000 | 0.719437 |
| 1.5 | 0.950000 | 0.948054 |
| 2.0 | 1.200000 | 1.176670 |

![Regression fit](results/part1/regression.png)

Each ReLU unit is active over part of the x-axis, and their weighted sum forms a piecewise-linear fit. Backpropagation and SGD reduce squared error on noisy samples. The model learns the overall trend but does not fully capture the breakpoints at `x=-0.4` and `x=0.5`; at `x=0` it predicts 0.259591 instead of 0.35.

### 4.2 Concentric-Ring Binary Classification

The model correctly classifies **799/800** training samples (99.875%); class 0 is the inner ring and class 1 is the outer ring.

![Concentric-ring classification boundary](results/part1/binary_boundary.png)

A single straight line cannot separate the concentric rings. Multiple ReLU activation regions combine into an approximately closed polygonal boundary that separates the inner and outer rings. High accuracy can coexist with a cross-entropy of 0.281615 because a correct prediction does not require the true class probability to be close to 1.

### 4.3 Three-Blob Multiclass Classification

The model correctly classifies **856/900** training samples (95.111%).

![Three-blob classification boundary](results/part1/multiclass_boundary.png)

ReLU units form multiple linear regions to separate the clusters. Three Softmax outputs provide class probabilities, and `argmax` selects the predicted class. Most independent-test errors involve the central blob and an outer blob; overlapping Gaussian distributions make perfect separation unlikely.

## 5. Verification and Reproduction

Previously executed numerical checks are recorded in [verification.json](results/part1/verification.json): maximum regression and classification gradient errors were `2.27e-9` and `9.45e-11`, and the Softmax stability check passed. These are archived results; the optional verification scripts have been removed. All three experiments have finite losses below their initial values.

Run from the project root; dependency installation is needed only for a fresh environment:

```powershell
python -m pip install --target .packages -r requirements.txt
python src/part1.py
```

Raw evidence is in [metrics.json](results/part1/metrics.json), [losses.csv](results/part1/losses.csv), [regression_predictions.csv](results/part1/regression_predictions.csv), and [run.log](results/part1/run.log). Optional test confusion matrices and numerical checks are also retained under `results/part1/`.

## 6. Responses to the Lab's Explanatory Points

The PDF's explanatory points are answered in the sections below, without repeating the same discussion as a separate question list.

| PDF topic | Response in this report |
| --- | --- |
| ANN, MLP, and the role of ReLU (pages 1-2) | Section 1: weighted sums, nonlinearity, architecture, initialization |
| Forward pass, losses, backpropagation, SGD (pages 2-9) | Sections 1-2: outputs, loss functions, gradients, mini-batches, updates |
| Example 1: piecewise-linear regression (page 9) | Section 4.1: nine predictions, ReLU fit, error and limitations |
| Example 2: nonlinear binary classification (pages 9-10) | Section 4.2: accuracy and how a nonlinear boundary separates rings |
| Example 3: three-class classification (page 10) | Section 4.3: accuracy, Softmax/argmax, cluster separation and errors |

## 7. Part I Completion Status

Part I's NumPy MLP and all three application examples are implemented, executed, and explained above. Cat/dog image processing belongs to Part II; GPU comparison belongs to Part III.

## 8. Part II: NumPy Cat/Dog CNN

### 8.1 Dataset, Model, and Necessary Corrections

The PDF's [microsoft/cats_vs_dogs dataset](https://huggingface.co/datasets/microsoft/cats_vs_dogs/tree/b5ae3589204019bc2cc97e99e4914a54589333ef) contains 23,410 Parquet rows. After removing 24 byte-identical duplicates, a seeded, stratified 80/20 split produces:

| Split | Cat (0) | Dog (1) | Total |
| --- | ---: | ---: | ---: |
| Training | 9,380 | 9,328 | 18,708 |
| Unseen test | 2,346 | 2,332 | 4,678 |

Images are organized under `data/train/{cat,dog}/` and `data/test/{cat,dog}/`, converted to RGB, resized to 64x64, and normalized to `[0,1]`. The model uses one fixed random 3x3x3 convolution filter, ReLU, 2x2 max pooling, and a scalar FC output with ReLU. Only FC weights and bias are trained, using MSE, online SGD, learning rate 0.001, and 10 epochs. The class is dog when the output exceeds 0.5, otherwise cat. Split seed is 42; model and training-order seed is 0.

Two corrections are necessary: valid convolution and pooling produce `64 -> 62 -> 31`, so the FC input is `31*31 = 961`, rather than the PDF's `32*32*3`. Also, splitting the PDF's class-ordered loader directly would leave only dogs in the test set; stratification ensures both classes appear in each split. Exact-byte duplicates are removed before splitting, and no test images are used for training or model selection.

### 8.2 PDF Task: Automatically Train and Test Large Blocks of Unseen Images

[src/part2.py](src/part2.py) automatically trains on all 18,708 training images, then tests all 4,678 held-out images once, in nine blocks of 512 and one final block of 70. Each block runs image loading, preprocessing, convolution, pooling, and FC prediction. The fixed training features are computed once and reused across epochs; this preserves the FC-only updates. The convolution filter is verified to remain unchanged.

### 8.3 PDF Task: Measure Image Processing Time

Wall-clock times are measured with `perf_counter()` on CPU (Windows 11, Python 3.13.15, NumPy 2.5.4):

| Measurement | Result |
| --- | ---: |
| Training image loading, resizing, convolution, pooling | 166.4118 s |
| FC training and epoch metrics, all 10 epochs | 3.0306 s |
| Total training processing | 169.4424 s |
| Processing all 4,678 test images | 39.4744 s |
| Test throughput | 118.5072 images/s |
| Average test time per image | 8.4383 ms |

Test timings include reading, RGB conversion, resizing, normalization, convolution, pooling, FC inference, and thresholding. They exclude downloading and result-file/plot export. Training time includes the one-time feature extraction disclosed above. Timings depend on machine load and filesystem caching.

### 8.4 PDF Task: Report the Relevant Results

| Metric | Result |
| --- | ---: |
| Initial full training MSE | 0.497860 |
| Final online training MSE (epoch 10) | 0.245325 |
| Final full training MSE | 0.244767 |
| Training accuracy | 60.4875% |
| Test MSE | 0.245212 |
| Test accuracy | 58.9996% (2,760 / 4,678) |
| Test majority-class baseline | 50.1496% |

The online MSE averages losses before individual SGD updates; the full training MSE evaluates the final epoch model on the entire training set. Test confusion matrix (rows = true class, columns = predicted class):

| True / Predicted | Cat (0) | Dog (1) |
| --- | ---: | ---: |
| Cat (0) | 1,315 | 1,031 |
| Dog (1) | 887 | 1,445 |

| Test block | Images | Accuracy | Processing time (s) |
| --- | ---: | ---: | ---: |
| 1 | 512 | 58.0078% | 4.3097 |
| 2 | 512 | 59.3750% | 4.2407 |
| 3 | 512 | 60.1562% | 4.5574 |
| 4 | 512 | 58.0078% | 4.2756 |
| 5 | 512 | 60.7422% | 4.1946 |
| 6 | 512 | 59.5703% | 4.3030 |
| 7 | 512 | 57.4219% | 4.2533 |
| 8 | 512 | 59.5703% | 4.1899 |
| 9 | 512 | 57.8125% | 4.4904 |
| 10 | 70 | 61.4286% | 0.6598 |

![Part II training loss and unseen-image block accuracy](results/part2/training_and_evaluation.png)

Training MSE decreases, and test accuracy is about 8.85 percentage points above the majority-class baseline. Accuracy remains modest: only the FC layer learns, while the single random convolution filter cannot learn task-specific cat/dog features. Output ReLU can also produce zero gradients for inactive outputs, although none of the final test outputs was zero. The training/test accuracy gap is about 1.49 percentage points. Results use the epoch-10 model without test-based checkpoint selection.

The three Part II evaluation tasks are complete. Raw evidence is in [metrics.json](results/part2/metrics.json), [losses.csv](results/part2/losses.csv), [test_blocks.csv](results/part2/test_blocks.csv), and [test_predictions.csv](results/part2/test_predictions.csv). Reproduce preparation and training from the project root:

```powershell
python scripts/prepare_part2_data.py
python src/part2.py
```
## 9. Part III: GPU Acceleration with PyTorch

### 9.1 Objective and Experimental Setup

Part III implements a PyTorch CNN for cat/dog classification and compares its recorded performance with the NumPy implementation from Part II.

The implementation is in [src/part3.py](src/part3.py). It reuses the same prepared dataset and train/test split described in Section 8.1: 18,708 training images and 4,678 unseen test images. Labels are cat = 0 and dog = 1.

`ImageFolder` loads the images, while the transformation pipeline resizes them to 64×64 using bicubic interpolation and converts them to float32 tensors with values in [0,1]. The training DataLoader shuffles samples each epoch; the test DataLoader does not shuffle.

| Setting | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 |
| Python | 3.10.10 |
| PyTorch | 2.7.1+cu128 |
| Device | CUDA |
| Model seed | 42 |
| Epochs | 10 |
| Training batch size | 64 |
| Test block size | 512 |
| DataLoader workers | 0 |
| Loss function | Mean squared error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Classification threshold | Output > 0.5 predicts dog; otherwise cat |

The model and each batch of images and labels are moved to the same device using `.to(device)`. Labels are converted to floating-point tensors with shape `[B, 1]` to match the model output.

### 9.2 CNN Architecture and Training

The model follows the detailed Part III instructions: two convolutional layers with ReLU and max pooling, followed by two fully connected layers with ReLU.

| Layer | Output shape |
| --- | --- |
| Input | B × 3 × 64 × 64 |
| Conv2d(3, 16, kernel=3, padding=1), ReLU | B × 16 × 64 × 64 |
| Max pooling, size 2 | B × 16 × 32 × 32 |
| Conv2d(16, 32, kernel=3, padding=1), ReLU | B × 32 × 32 × 32 |
| Max pooling, size 2 | B × 32 × 16 × 16 |
| Flatten | B × 8192 |
| Linear(8192, 64), ReLU | B × 64 |
| Linear(64, 1), ReLU | B × 1 |

Here, B is the actual batch size. The network has 529,505 trainable parameters, including biases. Unlike Part II, both convolutional and fully connected parameters are updated during training.

The final layer uses small normally distributed weights with standard deviation 0.001 and an initial bias of 0.5. This starts outputs near the classification threshold and reduces the risk of initially inactive output ReLUs. The output is a nonnegative classification score, not a probability, because ReLU does not constrain it to be at most 1.

Each training batch performs the following operations:

1. Clear gradients with `optimizer.zero_grad()`.
2. Compute predictions with `model(images)`.
3. Calculate MSE against the labels.
4. Compute gradients with `loss.backward()`.
5. Update parameters with `optimizer.step()`.

There are 293 batches per epoch and 2,930 parameter updates over 10 epochs. The final epoch-10 model is evaluated without test-based checkpoint selection.

Although the handout introduces Part III as the same classification system, its detailed instructions specify two convolutional layers, whereas Part II uses one fixed filter. Therefore, the implementations solve the same task but do not have identical architectures.

### 9.3 Training Results

Recorded training results are available in [losses.csv](results/part3/losses.csv).

| Epoch | Online training MSE | Training accuracy during updates | Time (s) |
| --- | ---: | ---: | ---: |
| 1 | 0.226647 | 62.13% | 102.67 |
| 2 | 0.195204 | 70.44% | 22.94 |
| 3 | 0.178759 | 73.09% | 23.07 |
| 4 | 0.161878 | 76.37% | 23.05 |
| 5 | 0.150008 | 78.44% | 22.68 |
| 6 | 0.139729 | 80.40% | 22.88 |
| 7 | 0.132218 | 81.36% | 23.01 |
| 8 | 0.124462 | 82.60% | 22.96 |
| 9 | 0.118859 | 83.71% | 22.96 |
| 10 | 0.113020 | 84.74% | 22.72 |

Training MSE decreased from 0.226647 to 0.113020, and training accuracy increased from 62.13% to 84.74%, indicating that the model learned useful classification features.

These training metrics are collected before each batch update while the model changes throughout the epoch. They are not a separate evaluation of the final model on the entire training set.

Total training time was 308.94 seconds. The first epoch took 102.67 seconds, while later epochs took approximately 23 seconds each. Initialization and filesystem caching may contribute to this difference, but their individual effects were not measured. The first epoch remains included in the total training time.

### 9.4 GPU Evaluation and Timing Method

After training, the code calls `model.eval()` and evaluates under `torch.no_grad()`. Evaluation does not update model parameters.

Before timing, three inference passes use a zero tensor of shape `[512, 3, 64, 64]` on the GPU. These warm-up passes are excluded from the reported test time.

Elapsed time is measured using `time.perf_counter()`. CUDA synchronization is performed at timing boundaries so pending GPU operations finish before elapsed times are recorded.

Test timing includes:

- Image loading and RGB conversion.
- Resizing and tensor conversion.
- CPU-to-GPU transfer.
- CNN inference.
- Classification and metric calculation.

It excludes dataset downloading, warm-up, and result-file export. Therefore, the reported throughput measures end-to-end image processing rather than GPU computation alone.

Training timing includes image loading and preprocessing during each epoch, data transfer, forward and backward computation, parameter updates, and metric/logging overhead. Training does not use a separate warm-up phase.

All 4,678 unseen images are evaluated in nine blocks of 512 images and a final block of 70.

| Test block | Images | Accuracy | Processing time (s) |
| --- | ---: | ---: | ---: |
| 1 | 512 | 81.05% | 2.3961 |
| 2 | 512 | 81.05% | 2.2572 |
| 3 | 512 | 79.30% | 2.3061 |
| 4 | 512 | 80.47% | 2.2071 |
| 5 | 512 | 82.62% | 2.2164 |
| 6 | 512 | 83.40% | 2.4077 |
| 7 | 512 | 80.08% | 2.2923 |
| 8 | 512 | 78.71% | 2.2433 |
| 9 | 512 | 80.27% | 2.2843 |
| 10 | 70 | 88.57% | 0.3084 |

Source: [test_blocks.csv](results/part3/test_blocks.csv).

Overall accuracy is calculated from the total number of correct predictions, rather than averaging block accuracies without weighting. The model correctly classified 3,784 out of 4,678 images.

Because `ImageFolder` groups images by class and the test loader does not shuffle, individual blocks are not necessarily class-balanced. Block accuracy differences should not be interpreted as changes in model quality over time. Block identities and ordering may also differ from Part II.

### 9.5 Comparison with the NumPy Implementation

| Metric | Part II: NumPy / CPU | Part III: PyTorch / GPU |
| --- | ---: | ---: |
| Test images | 4,678 | 4,678 |
| Test accuracy | 59.00% | 80.89% |
| Test MSE | 0.245212 | 0.139953 |
| Total training time | 169.44 s | 308.94 s |
| End-to-end test time | 39.47 s | 20.92 s |
| Test throughput | 118.51 images/s | 223.62 images/s |

Sources: [Part II metrics](results/part2/metrics.json) and [Part III metrics](results/part3/metrics.json).

![NumPy and PyTorch comparison](results/part3/comparison.png)

Test accuracy improved by 21.89 percentage points. End-to-end test time decreased by approximately 47%, and throughput was approximately 1.89 times the recorded NumPy throughput.

The improvement in accuracy is consistent with learning task-specific convolutional features. Part II trains only the fully connected layer on features extracted by one fixed random convolution filter, while Part III trains the entire CNN.

However, these results do not isolate the effect of GPU acceleration:

- Network architectures and trainable parameter counts differ.
- Part II uses online SGD and float64; Part III uses mini-batch Adam and float32.
- Part II computes fixed training features once and reuses them across epochs.
- Part III processes images and updates the full network during every epoch.
- The recorded experiments run on different computers.
- Timings come from individual recorded runs without repeated-run uncertainty estimates.

These differences explain why Part III can have higher accuracy and faster testing while taking longer to train. The training-time comparison does not establish that GPU training is slower for an equivalent workload.

A controlled GPU acceleration experiment would evaluate the same PyTorch model and weights on CPU and GPU on the same computer, using matched batch sizes, preprocessing, timing scope, and repeated measurements. This additional experiment was not performed.

### 9.6 Reproduction and Deliverables

Run the following commands from the project root in the configured environment:

powershell
.\.venv\Scripts\python.exe scripts/prepare_part2_data.py
.\.venv\Scripts\python.exe src/part3.py
.\.venv\Scripts\python.exe src/plot_comparison.py
The recorded run uses PyTorch 2.7.1+cu128 and torchvision 0.22.1. For a fresh environment, install the repository dependencies and the GPU packages:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

The script uses CUDA when available and otherwise falls back to CPU. Model seeds are fixed, but strict deterministic execution is not enabled. Timing and numerical results may vary across environments.
File	Purpose
[part3.py](src/part3.py)	CNN training, GPU evaluation, and result export
[plot_comparison.py](src/plot_comparison.py)	Generate comparison charts
[metrics.json](results/part3/metrics.json)	Configuration and aggregate results
[losses.csv](results/part3/losses.csv)	Per-epoch training statistics
[test_blocks.csv](results/part3/test_blocks.csv)	Per-block test statistics
[model.pt](results/part3/model.pt)	Trained model state dictionary
[comparison.png](results/part3/comparison.png)	Comparison figure
[comparison.pdf](results/part3/comparison.pdf)	PDF version of the comparison figure


The model state dictionary requires the matching CatDogCNN class when reloaded. Rerunning the scripts overwrites the corresponding Part III result files.
The Part III Video Log has been recorded separately to explain the implementation, training behavior, test results, and comparison limitations.
10. Conclusion
The laboratory progresses from a manually implemented NumPy MLP to image classification with a NumPy CNN and a GPU-enabled PyTorch CNN.
Part III automatically trains on the prepared training set, evaluates large blocks of unseen images, measures processing time, and saves the relevant results. The recorded model achieves 80.89% test accuracy and an end-to-end test throughput of 223.62 images per second.
Compared with the recorded Part II implementation, it achieves higher classification accuracy and faster testing. Differences in architecture, optimization, precision, feature reuse, and hardware mean that the full improvement cannot be attributed to GPU acceleration alone.
