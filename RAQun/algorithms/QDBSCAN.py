from RAQun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.circuits import InnerProduct, InnerProduct1R
from RAQun.utils import mmng

class QDBSCAN(Algorithm):
    """
    Quantum-hybrid DBSCAN clustering algorithm.

    Parameters
    ----------
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    minSamples : int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """
    def __init__(self, eps: float, minSamples: int) -> None:
        """
        Initializes the class variables.

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        minSamples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        """
        pass

    def fit(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Fits the model to the data.

        Parameters
        ----------
        X : NDArray[np.floating]
            Data matrix of shape (nSamples, nFeatures).

        Returns
        -------
        NDArray[np.floating]
            Array of shape (nSamples,) with cluster labels.
        """
        pass
