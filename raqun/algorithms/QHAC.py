from raqun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from raqun.utils import matGen
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

class QHAC(Algorithm):
    """
        Quantum-hybrid Agglomerative Hierarchical Clustering algorithm.

        Parameters
        ----------
        k : int
            Number of clusters.
        paramType : str
            Type of the parameters, must be 'data' or 'similarity'.
    """
    
    def __init__(self, k: int = 2, paramType: str = 'data') -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            k : int
                Number of clusters.
            paramType : str
                Type of the parameters, must be 'data' or 'similarity'.
        """
        self.paramType = paramType
        self.k = k
        if self.paramType != "data" and self.paramType != "similarity":
            raise ValueError("paramType must be 'data' or 'similarity'")
    #end __init__

    def fit(self, params: NDArray[np.floating], *link: str) -> NDArray[np.floating]:
        """
            Fits the model to the data.

            Parameters
            ----------
            params : NDArray[np.floating]
                If paramType is 'data', the matrix is of shape (nSamples, nFeatures).
                If paramType is 'similarity', the matrix is of shape (nSamples, nSamples).
            link : str
                Linkages to use in GAS.

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with cluster labels.
        """
        clusters = np.empty((len(link), params.shape[0]))
        if self.paramType == "data":
            params = matGen(params)
            
        for i, l in enumerate(link):
            hier = AgglomerativeClustering(
                n_clusters = self.k, 
                metric = 'precomputed', 
                linkage = l
            )
            cluster = hier.fit_predict(params)
            clusters[i] = cluster

        return clusters
    #end fit

# X = pd.read_csv("/home/elma/Documentos/raqun/instances.csv", delimiter=",").iloc[:, :-1].to_numpy()

# h = QHAC()

# labels = h.fit(X, 'single', 'complete', 'average')
# print(labels)
