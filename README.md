# Lab 7 - Part I

Part I of the PDF is complete, including the NumPy MLP and all three application examples. See [docs/report.md](docs/report.md) for the analysis.

```text
ee542lab7/
|-- README.md
|-- requirements.txt
|-- src/
|   `-- part1.py                 # Part I implementation and three experiments
|-- scripts/
|   `-- verify_part1.py          # Optional gradient and Softmax checks
|-- docs/
|   `-- report.md               # Part I report and explanations
`-- results/
    `-- part1/                  # Figures, metrics, CSV files, and execution log
```

Run these commands from the project root:

```powershell
python -m pip install --target .packages -r requirements.txt
python src/part1.py
python scripts/verify_part1.py
```

Dependencies are already installed in this project's `.packages/` directory, so the last two commands can be run directly. Both scripts resolve paths from their own locations; results are always saved to `results/part1/` at the project root.

- [src/part1.py](src/part1.py): MLP, SGD, regression, binary and multiclass classification, and result and figure export.
- [scripts/verify_part1.py](scripts/verify_part1.py): Optional finite-difference checks of backpropagation gradients and a Softmax stability check; not required by the PDF.
- `results/part1/`: Recorded experiment results and figures.

Network calculations and training use only NumPy. Matplotlib is used only for plotting. Parts II and III have not been implemented.

Local dependency and cache directories (`.packages/`, `.mplconfig/`, and `__pycache__/`) are excluded by `.gitignore`. They are not report deliverables. `.git/` contains local version history.
