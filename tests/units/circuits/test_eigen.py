import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pennylane as qml

from RAQun.circuits import Eigen

@pytest.fixture
def sampleMatrix():
    return np.array([[2.0, 1.0], [1.0, 2.0]])

def test_eigen_init(sampleMatrix):
    eigen = Eigen(sampleMatrix)
    
    assert hasattr(eigen, 'X')
    assert hasattr(eigen, 'norm')
    assert hasattr(eigen, 'n')
    assert hasattr(eigen, 'logN')
    assert hasattr(eigen, 'm')
    assert hasattr(eigen, 'logM')
    assert hasattr(eigen, 'Q')
    assert hasattr(eigen, 'regA')
    assert hasattr(eigen, 'regI')
    assert hasattr(eigen, 'qnode')
    
    expectedNorm = np.linalg.norm(sampleMatrix, ord=2)
    expectedX = (sampleMatrix / expectedNorm) + 0.5 * np.eye(2)
    np.testing.assert_allclose(eigen.X, expectedX, atol=1e-7)
    
    assert eigen.n == 2
    assert eigen.logN == 1
    assert eigen.m == 500
    assert eigen.logM == 9

def test_eigen_run(sampleMatrix):
    eigen = Eigen(sampleMatrix)
    res = eigen.run()
    assert isinstance(res, qml.measurements.MeasurementProcess)

def test_eigen_oracle(sampleMatrix):
    eigen = Eigen(sampleMatrix)
    wires = eigen.regI
    
    with qml.tape.QuantumTape() as tape:
        eigen.oracle(1, wires)
        
    assert len(tape.operations) == 1
    assert tape.operations[0].name == 'Exp'

def test_eigen_probs(sampleMatrix):
    eigen = Eigen(sampleMatrix)
    
    kTotal = eigen.logM + eigen.logN
    mockCounts = {
        '0' * kTotal: 512,
        ('0' * (kTotal - 1)) + '1': 512
    }
    
    eigen.qnode = MagicMock(return_value=mockCounts)
    
    probs = eigen.probs()
    
    assert isinstance(probs, dict)
    assert len(probs) == 2**kTotal
    assert probs['0' * kTotal] == 512
    assert probs[('0' * (kTotal - 1)) + '1'] == 512
    assert probs['1' * kTotal] == 0

def test_eigen_vectors(sampleMatrix):
    eigen = Eigen(sampleMatrix)
    
    kTotal = eigen.logM + eigen.logN
    mockProbsDict = {f"{i:0{kTotal}b}": 0 for i in range(2**kTotal)}
    mockProbsDict[f"{0:0{kTotal}b}"] = 512
    mockProbsDict[f"{1:0{kTotal}b}"] = 512
    
    with patch.object(eigen, 'probs', return_value=mockProbsDict):
        vecs = eigen.vectors(1)
        
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 2)