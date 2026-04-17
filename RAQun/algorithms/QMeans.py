from RAQun.algorithms.base import Algorithm
from RAQun.algorithms.QEuclidean import QEuclidean
import numpy as np
from numpy.typing import NDArray

class QMeans(Algorithm):
    """
        Quantum-hybrid K-Means clustering algorithm.

        Parameters
        ----------
        k : int
            Number of clusters.
        maxIters : int
            Maximum number of iterations.
    """
    
    def __init__(self, k: int, maxIters: int) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            k : int
                Number of clusters.
            maxIters : int
                Maximum number of iterations before ending.
        """
        pass
    #end __init__

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
    #end fit
