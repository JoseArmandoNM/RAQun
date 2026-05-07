from raqun.algorithms import QHAC
import numpy as np
from unittest.mock import patch

@patch('raqun.algorithms.QHAC.matGen')
def test_qhac(mock_matGen):
    mock_matGen.return_value = np.array([
        [0.0, 2.0, 2.0, 9.0, 9.0, 9.0],
        [2.0, 0.0, 4.0, 9.0, 9.0, 9.0],
        [2.0, 4.0, 0.0, 9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0, 0.0, 2.0, 2.0],
        [9.0, 9.0, 9.0, 2.0, 0.0, 4.0],
        [9.0, 9.0, 9.0, 2.0, 4.0, 0.0]
    ])
    
    qhac = QHAC(k=2, paramType='data')
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    labels = qhac.fit(X, 'single', 'complete')
    
    assert labels.shape == (2, 6)
    for linkage_labels in labels:
        assert linkage_labels[0] == linkage_labels[1] == linkage_labels[2]
        assert linkage_labels[3] == linkage_labels[4] == linkage_labels[5]
        assert linkage_labels[0] != linkage_labels[3]
