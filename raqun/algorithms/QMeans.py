from raqun.algorithms.base import Algorithm
from raqun.algorithms.QEuclidean import QEuclidean
import numpy as np
from numpy.typing import NDArray
from typing import List, Any

class QMeans(Algorithm):
    """
        Quantum-hybrid K-Means clustering algorithm.

        Parameters
        ----------
        k : int
            Number of clusters.
        maxIters : int
            Maximum number of iterations.
    """
    
    def __init__(self, k: int, maxIters: int = 50, metric: str = 'hadamard') -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            k : int
                Number of clusters.
            maxIters : int
                Maximum number of iterations before ending.
            metric : str
                Metric to use for the quantum circuit.
                Options: 'hadamard' or 'swap'.
        """

        self.k: int = k
        self.maxIters: int = maxIters
        self.history: List[NDArray[np.floating]] = [] 
        self.metric = metric.lower()
        if self.metric not in ['hadamard', 'quantum']:
            raise ValueError(f'Metric {metric} not recognized.')
    #end __init__

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
        n = X.shape[0]
        current_labels: NDArray[np.floating] = np.array(np.random.choice(self.k, size=n), dtype=np.float64)
        
        for i in range(self.maxIters):
            classifier: QEuclidean = QEuclidean(self.metric)
            classifier.fit(X, current_labels)
            newLabels: NDArray[np.floating] = np.empty(n)
            changes: int = 0

            for i, x in enumerate(X):
                predictedLabel: Any = classifier.predict(x)
                newLabels[i] = predictedLabel
                if predictedLabel != current_labels[i]:
                    changes += 1
            
            self.history.append(newLabels)
            current_labels = newLabels
            if changes == 0:
                break
        
        return current_labels
    #end fit
#end QMeans

'''
if __name__ == '__main__':
    import pandas as pd
    X = pd.read_csv("/home/elma/Documentos/raqun/dataPrueba.csv")
    X = X.drop(columns=['id', 'label']).values
    qmeans = QMeans(k=2)
    labels = qmeans.fit(X)
    print(labels)
'''