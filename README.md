# py-occam

Python port of the OCCAM (Organizational Complexity Computation and Modeling) variable-based reconstructability analysis software.

This is a pure Python implementation that matches the output of the original C++ code from https://github.com/percyd/occam.

## Installation
```pip install -e .```

## Usage
```python
from py_occam import Occam
from py_occam.utils import load_data

# Load data with YAML sidecar
data, varlist = load_data('data.tsv')

# Create OCCAM instance
occam = Occam(data, varlist)

# Do search
models = occam.search(width=3, levels=7)

# Fit specific model
fit_results = occam.fit("IV:ApZ:EdZ:CZ")