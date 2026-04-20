from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen, padder
# from pennylane.measurements import Expectation

class VQE(Circuit):
    """
        Variational Quantum Eigensolver (VQE) algorithm for finding the eigenvalues and eigenvectors of a matrix.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.
    """

    def __init__(self, X: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            X : NDArray[np.floating]
                Matrix of shape (nFeatures, nSamples) with the data.
        """
        self.X = (X + X.T)/2
        self.norm = np.linalg.norm(X, ord=2)
        self.X = self.X / self.norm
        self.X = self.padder(self.X)
        # self.X = self.X + 0.5 * np.eye(self.X.shape[0])
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if self.X[i][j] == 0:
                    self.X[i][j] = 1e-10
        self.shots = None
        self.H = qml.pauli_decompose(self.X)
        self.H.grouping_indices = None
        self.n, self.log_n = log_t(X.shape[0])
        self.m, self.log_m = log_t(8)
        self.Q = ctrlGen(self.n, self.log_n)
        # self.reg_A = list(range(self.log_m))
        self.reg_I = list(range(self.log_n))
        self.dev = qml.device(
            "default.qubit", 
            wires = len(self.reg_I)
        )
        self.qnode = qml.QNode(
            self.circuit, 
            self.dev, 
            shots=self.shots
        )
        self.max = np.max(self.X)
    #end __init__

    def run(self) -> float:
        """
            Calculates the expectation value of the measurement of the circuit.

            Returns
            -------
            float
                Expectation value of the measurement of the circuit.
        """
        pass
    #end run

    def getStateQnode(self, params: NDArray[np.floating]) -> NDArray[np.floating]:
        pass
    #end getStateQnode

    def vqe_opt(iter: int) -> NDArray[np.floating]:
        """
            Optimizes the circuit using the VQE algorithm.

            Parameters
            ----------
            iter : int
                Number of iterations.

            Returns
            -------
            NDArray[np.floating]
                Vector of shape (nFeatures,) with the optimal parameters.
        """
        pass
    #end vqe_opt
#end VQE
