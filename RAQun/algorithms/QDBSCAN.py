from RAQun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.circuits import InnerProduct, InnerProduct1R
from RAQun.utils.graph import mmng, matGen
import pandas as pd
from collections import deque

class QDBSCAN(Algorithm):
    """
        Quantum-hybrid DBSCAN clustering algorithm.

        Parameters
        ----------
        eps : float
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        minSamples : int
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """
    
    def __init__(self, eps: float, minSamples: int) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            eps : float
                The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            minSamples : int
                The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        """

        self.eps = eps
        self.minSamples = minSamples
    #end __init__

    def fit(self, params: NDArray[np.floating], paramsType: str = 'data') -> NDArray[np.floating]:
        """
            Fits the model to the data.

            Parameters
            ----------
            params : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).
            paramsType : str
                Type of the given matrix.
                Options: 'data', 'mmng', 'distances'.

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with cluster labels.
            
            Raises
            ------
            ValueError
                If paramsType is not 'data', 'mmng', or 'distances'.
        """

        if paramsType == 'data':
            dists = matGen(params)
            mmng_list = mmng(dists, eps=self.eps)
        elif paramsType == 'mmng':
            mmng_list = params
        elif paramsType == 'distances':
            mmng_list = mmng(params, eps=self.eps)
        else:
            raise ValueError(f'Params type {paramsType} not recognized.')

        visited = [False] * len(mmng_list)
        clusters = []
        labels = [-1] * len(mmng_list)

        for ix, vs in enumerate(mmng_list):
            if visited[ix]:
                continue
            visited[ix] = True

            if len(vs) >= self.minSamples:
                current = {ix}
                clusters.append(current)
                queue = deque(vs)

                while queue:
                    vec = queue.popleft()
                    current.add(vec)
                    
                    if vec >= len(visited):
                        print("aquí: ", vec, len(visited))
                    if visited[vec]:
                        continue
                    visited[vec] = True
                    
                    vs_in = mmng_list[vec]
                    if len(vs_in) >= self.minSamples:
                        queue.extend(vs_in)
        
        for id, cluster in enumerate(clusters):
            for vec in cluster:
                labels[vec] = id
        
        for _ in clusters:
            print(len(_))

        return labels
    #end fit
#end QDBSCAN

if __name__ == '__main__':
    X = pd.read_csv("/home/elma/Documentos/RAQun/instances.csv")
    X = X[:-1]


    mat = matGen(X.to_numpy())

    mmng = mmng(mat)

    print(f'La matriz de similitud es: \n{mat}\n\n\n\n\n\n')
    # print(f'El mmng es: \n{mmng}\n\n\n\n\n\n')

    pass
