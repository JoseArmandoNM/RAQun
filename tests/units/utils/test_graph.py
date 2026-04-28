from RAQun.utils.graph import mmng, inMat, matGen
import numpy as np
from unittest.mock import patch

def test_mmng():
    params = [[0.0, 0.6, 0.2],
              [0.6, 0.0, 0.4],
              [0.2, 0.4, 0.0]]
    res = mmng(params, eps=0.5)
    assert res == [[0, 2], [1, 2], [0, 1, 2]]
#end test_mmng

def test_inMat():
    graph = [[0, 2], [1, 2], [0, 1, 2]]
    res = inMat(graph)
    expected = np.zeros((3, 3))
    expected[0, 1] = 1
    expected[1, 2] = 1
    expected[2, 1] = -1
    expected[2, 2] = -1
    np.testing.assert_array_equal(res, expected)
#end test_inMat

@patch('RAQun.utils.graph.InnerProduct1R')
@patch('RAQun.utils.graph.probs')
def test_matGen(mock_probs, mock_InnerProduct1R):
    mock_circuit = mock_InnerProduct1R.return_value
    mock_circuit.qnode.return_value = {'00': 1}
    mock_circuit.norms2 = np.array([1.0, 2.0])
    
    mock_probs.return_value = np.array([0.5, 0.5])
    
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = matGen(X)
    
    expected = np.array([[4.0, 6.0], [4.0, 6.0]])
    np.testing.assert_array_equal(res, expected)
#end test_matGen
