from .base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.utils import mmng, inMat
from RAQun.circuits import InnerProduct, InnerProduct1R

class QSC(Algorithm):
    """
    Quantum-hybrid Spectral Clustering algorithm.

    Parameters
    ----------
    k : int
        Number of clusters.
    eps : float
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    """
    
    def __init__(self, k: int, eps: float) -> None:
        """
        Initializes the class variables.

        Parameters
        ----------
        k : int
            Number of clusters.
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
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
    
    def projection(self, B: NDArray[np.floating], isQuantum: bool = False) -> NDArray[np.floating]:
        """
        Calculates the Eigenvectors of the Laplacian of the given graph.

        Parameters
        ----------
        B : NDArray[np.floating]
            Data matrix of shape (nSamples, nSamples(nSamples-1)/2).
        isQuantum : bool
            Whether to use the quantum method to calculate de Eigenvectors.

        Returns
        -------
        NDArray[np.floating]
            Array of shape (nSamples, nSample) with cluster labels.
        """
        pass
        