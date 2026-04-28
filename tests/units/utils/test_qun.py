from RAQun.utils.qun import ctrlGen, padder
import numpy as np

def test_ctrlGen():
    res = ctrlGen(3, 2)
    assert res == ['00', '01', '10']
#end test_ctrlGen

def test_padder():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    res = padder(X)
    assert res.shape == (4, 3)
    
    expected = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [5.0, 6.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    np.testing.assert_array_equal(res, expected)
#end test_padder
