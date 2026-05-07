import pytest
import numpy as np
import pennylane as qml
from unittest.mock import patch, MagicMock

from raqun.circuits import VQE

@pytest.fixture
def sampleMatrix():
    return np.array([[2.0, 1.0], [1.0, 2.0]])

def test_vqe_init(sampleMatrix):
    vqe = VQE(sampleMatrix)
    
    assert hasattr(vqe, 'X')
    assert hasattr(vqe, 'H')
    assert hasattr(vqe, 'n')
    assert hasattr(vqe, 'logN')
    assert hasattr(vqe, 'm')
    assert hasattr(vqe, 'logM')
    assert hasattr(vqe, 'regI')
    assert hasattr(vqe, 'qnode')
    
    assert vqe.logN == 1
    assert len(vqe.regI) == 1
    
    expectedNorm = np.linalg.norm(sampleMatrix, ord=2)
    expectedX = sampleMatrix / expectedNorm
    np.testing.assert_allclose(vqe.X, expectedX, atol=1e-7)

def test_vqe_run(sampleMatrix):
    vqe = VQE(sampleMatrix)
    
    shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=vqe.logN)
    params = np.random.random(size=shape)
    
    res = vqe.qnode(params)
    
    assert isinstance(res, (float, np.ndarray, qml.numpy.tensor))
    if isinstance(res, np.ndarray):
        assert res.shape == ()

def test_vqe_get_state_qnode(sampleMatrix):
    vqe = VQE(sampleMatrix)
    
    shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=vqe.logN)
    params = np.random.random(size=shape)
    
    probs = vqe.getStateQnode(params)
    
    assert isinstance(probs, (np.ndarray, qml.numpy.tensor))
    assert len(probs) == 2**vqe.logN
    
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-6)

@patch('pennylane.GradientDescentOptimizer')
def test_vqe_opt(mock_optimizer_class, sampleMatrix):
    vqe = VQE(sampleMatrix)
    
    mock_opt = MagicMock()
    mock_optimizer_class.return_value = mock_opt
    
    def mock_step_and_cost(qnode, params):
        return params, 0.5
        
    mock_opt.step_and_cost.side_effect = mock_step_and_cost
    
    res = vqe.vqeOpt(iter=5)
    
    assert mock_opt.step_and_cost.call_count == 5
    assert isinstance(res, (np.ndarray, qml.numpy.tensor))
    assert len(res) == 2**vqe.logN
    np.testing.assert_allclose(np.sum(res), 1.0, atol=1e-6)