from raqun.algorithms.base import Algorithm
from raqun.circuits import InnerProduct, InnerProduct1R
from raqun.utils.maths import probs
import numpy as np
from numpy.typing import NDArray
from typing import List, Any

class QEuclidean(Algorithm):
    """
        Quantum-hybrid Euclidean Classifier.

        Parameters
        ----------
        metric : str
            Metric to use for the quantum circuit.
            Options are 'hadamard' or 'swap'.
    """

    def __init__(self, metric: str = "hadamard") -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            metric : str
                Metric to use for the quantum circuit.
                Options are 'hadamard' or 'swap'.
        """

        self.metric = metric.lower()
        if self.metric != "hadamard" and self.metric != "swap":
            raise ValueError("Metric not supported. Use 'hadamard' or 'swap'.")
    #end __init__

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> Any: # type: ignore[override]
        """
            Trains the model with the given matrix data and labels vector.

            Parameters
            ----------
            X : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).
            y : NDArray[np.floating]
                Labels of shape (nSamples).

            Returns
            -------
            None
        """
        
        self.X = X
        self.y = y

        self.Y = np.unique(self.y)
        centroids = np.array([
            self.X[self.y==c].mean(axis=0) for c in self.Y], 
            dtype=np.float64
        )
        normas = np.array([
            np.linalg.norm(c) ** 2 for c in centroids], 
            dtype=np.float64
        )

        self.centroids = centroids
        self.normas = normas
    #end fit

    def predict(self, vec: NDArray[np.floating]) -> Any:
        """
            Predicts the class of each sample in the data matrix.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with predicted class for each sample.
        """
        self.circuit = InnerProduct1R(vec, self.centroids) if self.metric == "hadamard" else InnerProduct(vec, self.centroids)
        
        probsVec = probs(self.circuit.qnode(), self.centroids.shape[0])
        
        dists: List[float] = []
        for i in range(self.centroids.shape[0]):
            Z: np.float64 = (self.normas[i] + np.linalg.norm(vec) ** 2)
            dist: np.float64 = 4 * Z * (1 - probsVec[i]) 
            dists.append(float(dist))

        dists_arr = np.array(dists)

        min_idx: np.int64 = np.argmin(dists_arr)
        return self.Y[min_idx]
    #end predict
#end QEuclidean


'''
if __name__ == '__main__':
    import pandas as pd
    X = pd.read_csv("/home/elma/Documentos/raqun/instances.csv")
    y = X.iloc[:, -1].to_numpy()
    X = X.iloc[:, :-1].to_numpy()

    model = QEuclidean(metric="swap")
    model.fit(X, y)

    for i in range(X.shape[0]):
        print(f"Label: {y[i]}, Predicted: {model.predict(X[i])}")
'''