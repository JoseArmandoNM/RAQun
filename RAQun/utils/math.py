from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

def log_2(num: int) -> int:
    """
    Calculates the ceiling of the base-2 logarithm of a number.

    Parameters
    ----------
    num : int
        Number to calculate the logarithm of.

    Returns
    -------
    int
        Ceiling of the base-2 logarithm of the number.
    """
    pass

def log_t(num: int) -> tuple[int, int]:
    """
    Calculates the base-2 logarithm of a number.

    Parameters
    ----------
    num : int
        Number to calculate the logarithm of.

    Returns
    -------
    int
        Ceiling of the base-2 logarithm of the number.
    int 
        Number given as a parameter
    """
    pass

def dist(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    """
    Calculates the Euclidean distance between two vectors.

    Parameters
    ----------
    p : NDArray[np.floating]
        First vector.
    q : NDArray[np.floating]
        Matrix of different vectors.

    Returns
    -------
    float
        Euclidean distance between the two vectors.
    """
    pass

def probs(counts: dict[str, int]) -> NDArray[np.floating]:
    """
    Calculates the probabilities of each element in the dictionary.

    Parameters
    ----------
    counts : dict[str, int]
        Dictionary of counts.

    Returns
    -------
    NDArray[np.floating]
        Probabilities of each element.
    """
    pass