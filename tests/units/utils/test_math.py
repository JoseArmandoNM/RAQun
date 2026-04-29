from RAQun.utils.maths import log_2, log_t, dist, probs
import numpy as np
import pytest

def test_log_2():
    assert log_2(1) == 1
    assert log_2(2) == 1
    assert log_2(3) == 2
    assert log_2(4) == 2
    assert log_2(5) == 3
    assert log_2(8) == 3
    assert log_2(16) == 4
#end test_log_2

def test_log_t():
    assert log_t(1) == (1, 1)
    assert log_t(2) == (2, 1)
    assert log_t(3) == (3, 2)
    assert log_t(4) == (4, 2)
    assert log_t(5) == (5, 3)
    assert log_t(8) == (8, 3)
    assert log_t(16) == (16, 4)
#end test_log_t

def test_dist_hadamard():
    p = np.array([0.5, 0.25])
    norms = np.array([1.0, 2.0, 3.0])
    res = dist(p, method='hadamard', vecs=norms)
    np.testing.assert_array_almost_equal(res, np.array([2.0, 3.0]))
#end test_dist_hadamard

def test_dist_swap():
    p = np.array([0.5, 0.25])
    norms = np.array([1.0, 2.0, 3.0])
    res = dist(p, method='swap', vecs=norms)
    np.testing.assert_array_almost_equal(res, np.array([6.0, 12.0]))
#end test_dist_swap

def test_dist_invalid_method():
    with pytest.raises(ValueError):
        dist(np.array([0.5]), method='invalid', vecs=np.array([1.0, 1.0]))
#end test_dist_invalid_method

def test_probs():
    counts = {'00': 10, '01': 20, '10': 30, '11': 40}
    res = probs(counts, 2)
    np.testing.assert_array_almost_equal(res, np.array([1/3, 3/7]))
#end test_probs
