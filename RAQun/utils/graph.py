from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

def matGen(probs: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Generates a disimilarity matrix with probabilities got from the inner product of the vectors.

    Parameters
    ----------
    probs : NDArray[np.floating]
        Vector of shape (nSamples,) with the probabilities.

    Returns
    -------
    NDArray[np.floating]
        Matrix of shape (nFeatures, nSamples) with distances.
    """
    pass

def mmng(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculates the matrix of minimum neighbors of the data.

    Parameters
    ----------
    X : list[list[int]]
        Matrix of shape (nSamples, nFeatures) with the data.

    Returns
    -------
    NDArray[np.floating]
        Matrix of shape (nSamples, nearestNeighbors) with the nearest neighbors of each sample.
    """
    pass

def inMat(graph: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculates the incidence matrix of the given graph.

    Parameters
    ----------
    graph : NDArray[np.floating]
        Matrix of shape (nSamples, nearestNeighbors) with the nearest neighbors of each sample.

    Returns
    -------
    NDArray[np.floating]
        Matrix of shape (nSamples, nSamples(nSamples-1)/2) with the incidence matrix.
    """
    pass