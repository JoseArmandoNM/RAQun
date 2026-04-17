from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen
# from pennylane.measurements import Expectation

class Eigen(Circuit):
    """
        Calculates the eigenvalues and eigenvectors of a matrix using the Quantum Phase Estimation algorithm.

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
    #end __init__

    def run(self) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """
        pass
    #end run

    def oracle(self) -> Any:
        """
            Oracle that implements the unitary operator U.

            Returns
            -------
            qml.QubitUnitary
                Oracle that implements the unitary operator U.
        """
        pass
    #end oracle
#end Eigen
    