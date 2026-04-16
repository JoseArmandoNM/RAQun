from RAQun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.utils import matGen

class QHAC(Algorithm):
    """
        Quantum-hybrid Agglomerative Hierarchical Clustering algorithm.

        Parameters
        ----------
        k : int
            Number of clusters.
        isSim : bool
            Whether to use parametes as a similarity matrix or data matrix.
    """
    
    def __init__(self, k: int, isSim: bool = False) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            k : int
                Number of clusters.
            isSim : bool
                Whether to use parametes as a similarity matrix or data matrix.
        """
        pass
    #end __init__

    def fit(self, X: NDArray[np.floating], *link: str) -> NDArray[np.floating]:
        """
            Fits the model to the data.

            Parameters
            ----------
            X : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).
            link : str
                Linkages to use in GAS.

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with cluster labels.
        """
        pass
    #end fit
