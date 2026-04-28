from RAQun.algorithms.base import Algorithm
from RAQun.algorithms.QEuclidean import QEuclidean
import numpy as np
from numpy.typing import NDArray
import pandas as pd

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
        """
        self.k: int = k
        self.maxIters: int = maxIters
        self.history: list = [] 
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
        current_labels = np.random.randint(0, self.k, size=n)
        
        for i in range(self.maxIters):
            classifier: QEuclidean = QEuclidean(self.metric)
            classifier.fit(X, current_labels)
            newLabels: NDArray[np.floating] = np.empty(n)
            changes: int = 0

            for i, x in enumerate(X):
                predictedLabel: np.floating = classifier.predict(x)
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

if __name__ == '__main__':
    X = pd.read_csv("/home/elma/Documentos/RAQun/dataPrueba.csv")
    X = X.drop(columns=['id', 'label']).values
    qmeans = QMeans(k=2)
    labels = qmeans.fit(X)
    print(labels)
