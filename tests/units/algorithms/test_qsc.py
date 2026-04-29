from RAQun.algorithms import QSC
import numpy as np
from unittest.mock import patch, MagicMock

@patch('RAQun.algorithms.QSC.QMeans')
@patch('RAQun.algorithms.QSC.matGen')
def test_qsc(mock_matGen, mock_QMeans_class):
    mock_matGen.return_value = np.array([
        [1.0, 0.8, 0.8, 0.1, 0.1, 0.1],
        [0.8, 1.0, 0.6, 0.1, 0.1, 0.1],
        [0.8, 0.6, 1.0, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 1.0, 0.8, 0.8],
        [0.1, 0.1, 0.1, 0.8, 1.0, 0.6],
        [0.1, 0.1, 0.1, 0.8, 0.6, 1.0]
    ])
    
    mock_qmeans_instance = MagicMock()
    mock_qmeans_instance.fit.return_value = np.array([0, 0, 0, 1, 1, 1])
    mock_QMeans_class.return_value = mock_qmeans_instance
    
    qsc = QSC(k=2, eps=0.5, paramType='data', eigen='classical')
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    labels = qsc.fit(X)
    
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))
    mock_qmeans_instance.fit.assert_called_once()
