from RAQun.algorithms import QMeans
import numpy as np
from unittest.mock import patch

@patch('RAQun.algorithms.QMeans.QEuclidean.predict')
def test_qmeans(mock_predict):
    mock_predict.side_effect = lambda x: 0 if x[0] < 5 else 1
    
    qmeans = QMeans(k=2, maxIters=5)
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    labels = qmeans.fit(X)
    
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))
