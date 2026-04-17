from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen

class InnerProduct(Circuit):
    """
        Calculates the inner product of two quantum states by using Swap Test.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.
        y : NDArray[np.floating]
            Vector of shape (nFeatures,) with the label.
    """

    def __init__(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            X : NDArray[np.floating]
                Matrix of shape (nFeatures, nSamples) with the data.
            y : NDArray[np.floating]
                Vector of shape (nFeatures,) with the label.
        """
        pass
    #end __init__

    def nQubitsCalc(self) -> int:
        """
            Calculates the number of qubits required for the circuit.

            Returns
            -------
            int
                Number of qubits required by the circuit.
        """
        pass
    #end nQubitsCalc

    def normalize(self, vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """
            Normalizes a vector.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector to normalize.

            Returns
            -------
            NDArray[np.floating]
                Normalized vector.
        """
        pass
    #end normalize

    def initialize(self, C: NDArray[np.floating]) -> None:
        """
            Initializes the quantum states.

            Parameters
            ----------
            C : NDArray[np.floating]
                Vector of shape (nFeatures,) with the label.
        """
        pass
    #end initialize

    def run(self, vec: NDArray[np.floating]) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector of shape (nFeatures).

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """
        pass
    #end run
#end InnerProduct
