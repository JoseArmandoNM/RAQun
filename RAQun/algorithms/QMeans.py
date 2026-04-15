from RAQun.algorithms.base import Algorithm
from RAQun.algorithms.QDBSCAN import QDBSCAN
import numpy as np
from numpy.typing import NDArray

class QMeans(Algorithm):
    """
    Quantum-inspired K-Means clustering algorithm.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
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

    def fit(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Fits the model to the data.

        Parameters
        ----------
        X : NDArray[np.floating]
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.floating]
            Array of shape (n_samples,) with cluster labels.
        """
        pass




# """
# <COMENTARIO>

# Parameters
# ----------
# <PARÁMETRO> : <TYPE>
#     <COMENTARIO>

# Returns
# -------
# <TYPE>
#     <COMENTARIO>
# """