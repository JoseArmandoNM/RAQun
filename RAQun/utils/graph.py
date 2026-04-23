from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

from RAQun.circuits.InnerProduct1R import InnerProduct1R


def mmng(params: NDArray[np.floating]) -> NDArray[np.floating]:
    """
        Calculates the matrix of minimum neighbors of the data.

        Parameters
        ----------
        params : list[list[int]]
            Matrix of shape (nSamples, nFeatures) with the data.

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (nSamples, nearestNeighbors) with the nearest neighbors of each sample.
    """
    mmng = []
    for vec in params:
        vec_aux = []
        for ix, val in enumerate(vec):
            if val <= self.eps:
                vec_aux.append(ix)
        mmng.append(vec_aux)

    return np.array(mmng)
#end mmng

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
    
    m = len(graph)
    in_matrix = np.zeros((m, int((m-1)*m/2)))
    for i in range(len(graph)):
        for j in graph[i]:
            if i == j:
                continue
            elif i < j:
                pos = int(i*(2*self.m-i-3)/2 + j - 1)
                val = 1
            else:
                pos = int(j*(2*self.m-j-3)/2 + i - 1)
                val = -1

            in_matrix[i][pos] = val
    
    return in_matrix
#end inMat


def matGen(X: NDArray[np.floating]) -> NDArray[np.floating]:
    pass
#end matGen

'''
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
            Matrix of shape (nFeatures, nSamples) with distances.
    """

    circuit = InnerProduct1R()
    N: np.int64 = len(X)
    mat: NDArray[np.floating] = np.zeros((N, N))

    for i, rec in enumerate(X):
        counts = circuit.run(rec, X)
        probs = probs(counts, N)
        dists = dist(probs)
        mat[i] = dists

    return mat
#end matGen
'''