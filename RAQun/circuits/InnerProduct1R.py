from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.math import log_t, probs
from RAQun.utils.qun import ctrlGen

class InnerProduct1R(Circuit):
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

    def run(self, vec: NDArray[np.floating]) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector of shape (nFeatures,).

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """
        pass