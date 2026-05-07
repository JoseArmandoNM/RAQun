from raqun.algorithms import QDBSCAN
import numpy as np
from unittest.mock import patch

@patch('raqun.algorithms.QDBSCAN.matGen')
def test_qdbscan(mock_matGen):
    mock_matGen.return_value = np.array([
        [0, 2, 2, 9, 9, 9],
        [2, 0, 4, 9, 9, 9],
        [2, 4, 0, 9, 9, 9],
        [9, 9, 9, 0, 2, 2],
        [9, 9, 9, 2, 0, 4],
        [9, 9, 9, 2, 4, 0]
    ])
    
    qdbscan = QDBSCAN(eps=2.5, minSamples=2)
    labels = qdbscan.fit(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]))    
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))