from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.math import log_t, probs
from RAQun.qun import ctrlGen
from pennylane.measurements import Expectation

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
        pass

    def run(self) -> float:
        """
        Calculates the expectation value of the measurement of the circuit.

        Returns
        -------
        float
            Expectation value of the measurement of the circuit.
        """
        pass

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