from raqun.circuits.InnerProduct1R import InnerProduct1R
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
    circ = InnerProduct1R(X[0], X)
    assert circ.vec.shape[0] == X.shape[1]
    assert circ.vecs.shape[0] == X.shape[0]
    assert circ.vecs.shape[1] == X.shape[1]
    assert circ.logK == int(np.ceil(np.log2(X.shape[0])))
    assert circ.n, circ.logN == log_t(X.shape[0])
    assert len(circ.Q) == X.shape[0]
    assert len(circ.Q[0]) == int(np.ceil(np.log2(X.shape[0])))
    assert len(circ.dev.wires) == int(np.ceil(np.log2(X.shape[0]))) + int(np.ceil(np.log2(X.shape[1]))) + 2
    assert circ.qnode.shots.total_shots == 1024

def test_inner_product_run(sampleData):
    X, y = sampleData
    circ = InnerProduct1R(X[0], X)
    res = circ.qnode()
    assert isinstance(res, dict)
    assert len(res) > 0
    assert len(list(res.keys())[0]) == circ.logK + 1
