from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

def ctrlGen() -> NDArray[np.str_]:
    """
    Generates a vector of control strings for the Q-UN algorithm.

    Returns
    -------
    NDArray[np.str_]
        Vector of control strings.
    """
    pass

def padder(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Pads the given matrix with zeros to make it a square (2^k, 2^K) matrix.

    Parameters
    ----------
    X : NDArray[np.floating]
        Matrix of shape (nSamples, nFeatures) with the data.

    Returns
    -------
    NDArray[np.floating]
        Matrix of shape (2^k, 2^K) with the data padded with zeros.
    """
    pass

