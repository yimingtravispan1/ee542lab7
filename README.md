# Lab 7 - NumPy Neural Networks

Parts I and II of the PDF are complete. The main [report.md](report.md) contains the Part I analysis and concise responses to all three Part II evaluation tasks in Section 8.

Part II implements the cat/dog NumPy CNN, training only the fully connected layer, with timed evaluation on held-out image blocks.

```text
ee542lab7/
|-- README.md
|-- requirements.txt
|-- src/
|   |-- part1.py                 # Part I implementation and three experiments
|   `-- part2.py                 # Part II CNN, training, timed block evaluation
|-- scripts/
|   `-- prepare_part2_data.py    # Parquet download, extraction, held-out split
|-- report.md                   # Main report: Parts I and II
|-- results/
|   |-- part1/                  # Part I recorded outputs
|   `-- part2/                  # Part II model, predictions, timing, and figures
`-- data/                       # Local dataset, ignored by Git
    |-- raw/                    # Downloaded Parquet shards
    |-- manifest.json           # Source rows, image hashes, labels, split paths
    |-- train/
    |   |-- cat/
    |   `-- dog/
    `-- test/
        |-- cat/
        `-- dog/
```

Run these commands from the project root:

```powershell
python -m pip install --target .packages -r requirements.txt
python src/part1.py
```

Dependencies are already installed in this project's `.packages/` directory, so the training command can be run directly. The scripts resolve paths from their own locations; results are saved under `results/` at the project root.

Run Part II from the project root:

```powershell
python scripts/prepare_part2_data.py
python src/part2.py
```

Preparation downloads approximately 722 MB from the PDF's Hugging Face dataset, verifies both shard hashes, removes byte-identical duplicates, and makes a deterministic stratified 80/20 split. It reuses a complete existing split. The local `data/` directory contains the full dataset and is excluded from Git.

Part II defaults to 10 epochs, learning rate 0.001, online SGD, seed 0, and test blocks of 512 images. Only FC weights and bias are trained. Frozen convolution/pooling features are computed once for training and reused across epochs. Test processing time includes loading, resizing, convolution, pooling, and FC inference. To change evaluation block size:

```powershell
python src/part2.py --block-size 1024
```

Rerunning training updates Part II outputs. Recorded report timings describe the saved run and can vary on another run or machine.

- [src/part1.py](src/part1.py): MLP, SGD, regression, binary and multiclass classification, and result and figure export.
- `results/part1/`: Recorded experiment results and figures.

Network calculations and training use only NumPy. Pillow handles images, PyArrow reads Parquet, and Matplotlib creates figures. Part III has not been implemented.

Local dependency and cache directories (`.packages/`, `.mplconfig/`, and `__pycache__/`) are excluded by `.gitignore`. They are not report deliverables. `.git/` contains local version history.
