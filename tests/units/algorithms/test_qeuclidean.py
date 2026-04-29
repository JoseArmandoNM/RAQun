from RAQun.algorithms import QEuclidean
import numpy as np
from unittest.mock import patch

@patch('RAQun.algorithms.QEuclidean.InnerProduct1R')
@patch('RAQun.algorithms.QEuclidean.probs')
def test_qeuclidean(mock_probs, mock_InnerProduct1R):
    model = QEuclidean(metric="hadamard")
    
    X = np.array([[1.0, 1.0], [1.0, 2.0], [9.0, 9.0], [9.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    mock_probs.return_value = np.array([0.9, 0.1])
    pred = model.predict(np.array([1.0, 1.2]))
    assert pred == 0
    mock_probs.assert_called_once()
