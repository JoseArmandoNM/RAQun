from RAQun.algorithms.base import Algorithm
import numpy as np
from numpy.typing import NDArray
from RAQun.circuits import InnerProduct, InnerProduct1R
from RAQun.utils.graph import mmng, matGen
import pandas as pd

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
            is_mmng : bool
                Whether the given matrix is a min-max neighborhood matrix.

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with cluster labels.
        """
        if paramsType == 'data':
            dists = matGen(params)
            mmng = mmng(dists)
        elif paramsType == 'mmng':
            mmng = params
        elif paramsType == 'distances':
            mmng = mmng(params)
        else:
            raise ValueError(f'Params type {paramsType} not recognized.')

        visited = [False] * len(mmng)
        clusters = []
        labels = [-1] * len(mmng)

        for ix, vs in enumerate(mmng):
            if visited[ix]:
                continue
            visited[ix] = True

            if len(vs) >= self.min_pts:
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
                    
                    vs_in = mmng[vec]
                    if len(vs_in) >= self.min_pts:
                        queue.extend(vs_in)
        
        for id, cluster in enumerate(clusters):
            for vec in cluster:
                labels[vec] = id
        
        for _ in clusters:
            print(len(_))

        return labels
    #end fit
#end QDBSCAN

X = pd.read_csv('/home/elma/Documentos/RAQun/instances.csv')
X = X[:-1]


#mat = matGen(X.to_numpy())

# mmng = mmng(mat)

# print(f'La matriz de similitud es: \n{mat}\n\n\n\n\n\n')
# print(f'El mmng es: \n{mmng}\n\n\n\n\n\n')