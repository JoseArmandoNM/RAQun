# RAQun 🦝

**RAQun** is a Package that implements QML algorithms using [PennyLane](https://pennylane.ai/) and [scikit-learn](https://scikit-learn.org/).
It provides a unified en intuitive interface to test and evaluate some QML algorithms for clustering and classification tasks.

## Features

- **Quantum Algorithms**:
  - `QDBSCAN`: Quantum Density-Based Spatial Clustering of Applications with Noise.
  - `QEuclidean`: Quantum Euclidean Distance based Classifier.
  - `QHAC`: Quantum Hierarchical Agglomerative Clustering.
  - `QMeans`: Quantum K-Means Clustering.
  - `QSC`: Quantum Spectral Clustering.
- **Quantum Components**: Essential components for QML, including `Eigen`, `InnerProduct`, `InnerProduct1R` and `VQE`.

## Requirements

- Python >= 3.10
- `pennylane` >= 0.44
- `pennylane-lightning` >= 0.44
- `scikit-learn` >= 1.8
- `numpy` >= 2.4

## Installation

You can intall this package from source code. Cloning the repo and installing it using pip:

```bash
git clone https://github.com/JoseArmandoNM/RAQun
cd RAQun
pip install .
```

*Note: Use `pip install -e .` if you plan to modify the source code of the package (editable mode).*

## Quick Start Guide

In the following sections, we show how to use the elements of RAQun.

### Utils

RAQun provides a set of utility functions to facilitate the implementation of QML algorithms.

The following examples show how to use the utility functions:

#### Maths Functions

```python
from raqun.utils.maths import *
print(log_t(5))

counts = {'00': 524, '01': 200, '10': 112, '11': 112}

probsArr = probs(counts, 4)

dists = dist(probsArr)

print(dists)
```

#### Qun Functions

```python
from raqun.utils.qun import *

ctrl = ctrlGen(5, 3)
print(ctrl)

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat = np.array(mat)
paddedMat = padder(mat)
print(paddedMat)
```

### Circuits

The following examples show how to use the quantum circuits provided by RAQun:

#### Inner Product Circuits

`InnerProduct` (Swap Test) and `InnerProduct1R` (Hadamard Test) calculate the inner product/distance between a single vector and a matrix of vectors.

```python
from raqun.circuits import InnerProduct, InnerProduct1R
import numpy as np

vec = np.array([1, 2])
vecs = np.array([[1, 2], [2, 1], [0, 1]])

circ_swap = InnerProduct(vec, vecs)
counts_swap = circ_swap.qnode()
print("Swap Test counts:", counts_swap)

circ_hadamard = InnerProduct1R(vec, vecs)
counts_hadamard = circ_hadamard.qnode()
print("Hadamard Test counts:", counts_hadamard)
```

#### Eigen

The `Eigen` circuit uses Quantum Phase Estimation (QPE) to find the eigenvectors of a matrix.

```python
from raqun.circuits import Eigen
import numpy as np

X = np.array([[1.0, 0.5], [0.5, 1.0]])

eigen = Eigen(X)
eigenvectors = eigen.vectors(k=1)
print("Eigenvectors from QPE:", eigenvectors)
```

#### VQE

The `VQE` circuit uses the Variational Quantum Eigensolver algorithm to find eigenvectors.

```python
from raqun.circuits import VQE
import numpy as np

X = np.array([[1.0, 0.5], [0.5, 1.0]])

vqe = VQE(X)
optimal_vector = vqe.vqeOpt(iter=20)
print("Optimal state from VQE:", optimal_vector)
```

The following example shows how to use `QDBSCAN` with the RAQun package:

```python
import numpy as np
from raqun.algorithms import QDBSCAN

# Create some test data
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Initialize and fit the QDBSCAN algorithm
qdbscan = QDBSCAN(eps=1.0, minSamples=2)
labels = qdbscan.fit(X)

print("Cluster labels:", labels)
```

### Algorithms

The following examples show how to use the algorithms provided by RAQun:

#### QDBSCAN

The `QDBSCAN` algorithm is a quantum version of the Density-Based Spatial Clustering of Applications with Noise algorithm.

```python
import numpy as np
from raqun.algorithms import QDBSCAN

# Create some test data
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Initialize and fit the QDBSCAN algorithm
qdbscan = QDBSCAN(eps=1.0, minSamples=2)
labels = qdbscan.fit(X)

print("Cluster labels:", labels)
```

#### QEuclidean

The `QEuclidean` algorithm is a quantum-hybrid Euclidean distance-based classifier.

```python
from raqun.algorithms import QEuclidean
import numpy as np

# Training data and labels
X_train = np.array([[1, 2], [10, 10], [2, 1], [11, 12]])
y_train = np.array([0, 1, 0, 1])

# Initialize and train the QEuclidean model
model = QEuclidean(metric="hadamard")
model.fit(X_train, y_train)

# Predict a new sample
new_sample = np.array([2, 2])
prediction = model.predict(new_sample)
print("Predicted class:", prediction)
```

#### QHAC

The `QHAC` algorithm is a quantum-hybrid Agglomerative Hierarchical Clustering algorithm.

```python
from raqun.algorithms import QHAC
import numpy as np

X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])

# Initialize the model for 2 clusters
qhac = QHAC(k=2, paramType='data')

# Fit and return labels using different linkages
labels = qhac.fit(X, 'single', 'complete')
print("Labels for single and complete linkages:\\n", labels)
```

#### QMeans

The `QMeans` algorithm is a quantum-hybrid K-Means clustering algorithm.

```python
from raqun.algorithms import QMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])

# Initialize for 2 clusters with a maximum of 50 iterations
qmeans = QMeans(k=2, maxIters=50, metric='hadamard')

labels = qmeans.fit(X)
print("Cluster labels:", labels)
```

#### QSC

The `QSC` algorithm is a quantum-hybrid Spectral Clustering algorithm.

```python
from raqun.algorithms import QSC
import numpy as np

X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])

# Initialize for 2 clusters and neighborhood distance (eps) of 2.0
qsc = QSC(k=2, eps=2.0, paramType='data', eigen='classical')

labels = qsc.fit(X)
print("Cluster labels:", labels)
```

## Desarrollo y Pruebas (Testing)

Para ejecutar el conjunto de pruebas y asegurarte de que todo funciona correctamente, puedes usar `pytest`:

```bash
pip install pytest
pytest tests/
```

## Authors

- **José Armando Noguez Martínez** - [armandonoguez113@gmail.com](mailto:armandonoguez113@gmail.com)
- **Dra. Guohua Sun** - [gsun@cic.ipn.mx](mailto:gsun@cic.ipn.mx)

## License

This project is licensed under the terms of the license provided in the `LICENSE` file.
