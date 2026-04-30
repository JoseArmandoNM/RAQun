from typing import Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

from RAQun.circuits.InnerProduct1R import InnerProduct1R
from RAQun.utils.maths import probs, dist


def mmng(params: NDArray[np.floating], eps: float = 0.5) -> List[List[int]]:
    """
        Calculates the matrix of minimum neighbors of the data.

        Parameters
        ----------
        params : list[list[int]]
            Matrix of shape (nSamples, nFeatures) with the data.
        eps : float
            Maximum distance between two samples for one to be considered as in the neighborhood of the other.
            Its default value is 0.5.

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (nSamples, nearestNeighbors) with the nearest neighbors of each sample.
    """

    mmng: List[List[int]] = []
    for vec in params:
        vec_aux: List[int] = []
        for ix, val in enumerate(vec):
            if val <= eps:
                vec_aux.append(ix)
        mmng.append(vec_aux)

    return mmng
#end mmng

def inMat(graph: List[List[int]]) -> NDArray[np.floating]:
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
    
    m = len(graph)
    inMatrix = np.zeros((m, int((m-1)*m/2)))
    for i in range(len(graph)):
        for j in graph[i]:
            if i == j:
                continue
            elif i < j:
                pos = int(i*(2*m-i-3)/2 + j - 1)
                val = 1
            else:
                pos = int(j*(2*m-j-3)/2 + i - 1)
                val = -1

            inMatrix[i][pos] = val
    
    return inMatrix
#end inMat

def matGen(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
        Generates a disimilarity matrix with probabilities got from the inner product of the vectors.

        Parameters
        ----------
        X : NDArray[np.floating]
            Data matrix of shape (nSamples, nFeatures).

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (nSamples, nSamples) with distances.
    """

    N: int = len(X)
    mat: NDArray[np.floating] = np.zeros((N, N))
    
    norms2: NDArray[np.floating] = np.array([np.linalg.norm(x)**2 for x in X])

    for i, rec in enumerate(X):
        circuit = InnerProduct1R(rec, X)
        counts: Dict[str, Any] = circuit.qnode()
        p_vec: NDArray[Any] = probs(counts, N)
        
        dists: NDArray[np.floating] = np.array([])
        for j in range(N):
            Z: float = norms2[j] + np.linalg.norm(rec)**2
            dist_val: float = 4 * Z * (1 - p_vec[j])
            dists = np.append(dists, dist_val)
            
        mat[i] = dists

    return mat
#end matGen