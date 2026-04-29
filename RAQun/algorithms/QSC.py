from RAQun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.utils import mmng, inMat, matGen
from RAQun.circuits import Eigen, VQE
from RAQun.algorithms import QMeans
import pandas as pd
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
    
    def __init__(self, k: int, eps: float, paramType: str = 'data', eigen: str = 'classical') -> None:
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
        """

        self.k = k
        self.eps = eps
        paramOptions: list = ['data', 'mmng', 'similarity', 'inMatrix']
        if paramType not in paramOptions:
            raise ValueError(f"paramType must be one of {paramOptions}")
        self.paramType = paramType
        eigOptions: list = ['classical', 'qpe', 'vqe']
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
            sim = matGen(params)
            graph = mmng(sim, self.eps)
            self.B = inMat(graph)
        elif self.paramType == 'similarity':
            graph = mmng(params, self.eps)
            self.B = inMat(graph)
        elif self.paramType == 'mmng':
            self.B = inMat(params)
        elif self.paramType == 'inMatrix':
            self.B = params
        
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
        qmeans = QMeans(self.k, 35)
        return qmeans.fit(self.projection(X))
    #end fit
#end QSC


if __name__ == '__main__':
    X = pd.read_csv(Path(__file__).resolve().parents[2] / "dataPrueba.csv")
    X = X.drop(columns=['id','label']).values
    qsc = QSC(k=2, eps=1)
    labels = qsc.fit(X)
    print(labels)
