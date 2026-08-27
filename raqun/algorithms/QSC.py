from raqun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from raqun.utils import mmng, inMat, matGen
from raqun.circuits import Eigen, VQE
from raqun.algorithms.QMeans import QMeans
import pandas as pd
from typing import List, cast
from pathlib import Path

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
    
    def __init__(self, k: int, eps: float, paramType: str = 'data', eigen: str = 'classical', maxIters: int = 50, metric: str = 'hadamard', shots: int | None = None) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            k : int
                Number of clusters.
            eps : float
                The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            paramType : str
                Type of the given matrix.
                Options: 'data', 'mmng', 'similarity', 'inMatrix'.
            eigen : str
                Method to calculate the Eigenvectors.
                Options: 'classical', 'qpe', 'vqe'.
            maxIters : int
                Maximum iterations for the embedded QMeans.
            metric : str
                Quantum distance metric ('hadamard' or 'swap').
            shots : int | None
                Number of measurement shots.
        """

        self.k = k
        self.eps = eps
        self.maxIters = maxIters
        self.metric = metric
        self.shots = shots
        paramOptions: List[str] = ['data', 'mmng', 'similarity', 'inMatrix']
        if paramType not in paramOptions:
            raise ValueError(f"paramType must be one of {paramOptions}")
        self.paramType = paramType
        eigOptions: List[str] = ['classical', 'qpe', 'vqe']
        if eigen not in eigOptions:
            raise ValueError(f"eigen must be one of {eigOptions}")
        self.eigen = eigen
    #end __init__
    
    def projection(self, params: list[list[int]] | NDArray[np.floating]) -> NDArray[np.floating]:
        """
            Calculates the Eigenvectors of the Laplacian of the given graph.

            Parameters
            ----------
            params : list[list[int]] | NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures) or (nSamples, nSamples).
            
            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples, nSample) with cluster labels.
        """
        self.m = len(params)
        if self.paramType == 'data':
            sim = matGen(cast(NDArray[np.floating], params))
            graph = mmng(sim, self.eps)
            self.B = inMat(graph)
        elif self.paramType == 'similarity':
            graph = mmng(cast(NDArray[np.floating], params), self.eps)
            self.B = inMat(graph)
        elif self.paramType == 'mmng':
            self.B = inMat(cast(List[List[int]], params))
        elif self.paramType == 'inMatrix':
            self.B = cast(NDArray[np.floating], params)
        
        L = self.B @ self.B.T
        if self.eigen == 'qpe':
            vecs = Eigen(L).vectors(self.k)
        elif self.eigen == 'vqe':
            vecs = VQE(L).vqeOpt()
        else:
            vals, vecs = np.linalg.eigh(L)
            vecs = vecs[:, :self.k]
            
        return vecs
    #end projection

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
        self.qmeans = QMeans(self.k, maxIters=self.maxIters, metric=self.metric, shots=self.shots)
        return self.qmeans.fit(self.projection(X))
    #end fit
#end QSC


if __name__ == '__main__':
    X = pd.read_csv(Path(__file__).resolve().parents[2] / "dataPrueba.csv")
    X = X.drop(columns=['id','label']).values
    qsc = QSC(k=2, eps=1)
    labels = qsc.fit(X)
    print(labels)
