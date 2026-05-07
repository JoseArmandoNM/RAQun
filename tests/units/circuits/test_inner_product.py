from raqun.circuits.InnerProduct import InnerProduct
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sampleData():
    df = pd.read_csv('tests/dataPrueba.csv')
    X = df.iloc[:, 1:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y


def test_inner_product_init(sampleData):
    X, y = sampleData
    circ = InnerProduct(X[0], X)
    assert circ.vec.shape[0] == X.shape[1]
    assert circ.vecs.shape[0] == X.shape[0]
    assert circ.vecs.shape[1] == X.shape[1]
    assert circ.logC == int(np.ceil(np.log2(X.shape[0])))
    assert circ.d, circ.logD == log_t(X.shape[1])
    assert len(circ.Q) == X.shape[0]
    assert len(circ.Q[0]) == int(np.ceil(np.log2(X.shape[0])))
    assert len(circ.dev.wires) == int(np.ceil(np.log2(X.shape[0]))) + 2 * int(np.ceil(np.log2(X.shape[1]))) + 1
    assert circ.qnode.shots.total_shots == 10024

def test_inner_product_run(sampleData):
    X, y = sampleData
    circ = InnerProduct(X[0], X)
    res = circ.qnode()
    assert isinstance(res, dict)
    assert len(res) > 0
    assert len(list(res.keys())[0]) == circ.logC + 1
