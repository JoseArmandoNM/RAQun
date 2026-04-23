from RAQun.algorithms.base import Algorithm
from RAQun.circuits import InnerProduct, InnerProduct1R
from RAQun.utils.maths import probs
import numpy as np
import pandas as pd

class QEuclidean(Algorithm):
    """
        Quantum-hybrid Euclidean Classifier.

        Parameters
        ----------
        X : NDArray[np.floating]
            Data matrix of shape (nSamples, nFeatures).
        y : NDArray[np.floating]
            Labels of shape (nSamples).
    """

    def __init__(self, metric: str = "hadamard") -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            metric : str
                Metric to use for the quantum circuit.
                Options: 'hadamard' or 'swap'.
        """
        self.metric = metric.lower()
        if self.metric != "hadamard" and self.metric != "swap":
            raise ValueError("Metric not supported. Use 'hadamard' or 'swap'.")
    #end __init__

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
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
        self.circuit = InnerProduct1R(self.X, self.y) if self.metric == "hadamard" else InnerProduct(self.X, self.y)
    #end train

    def predict(self, vec: NDArray[np.floating]) -> NDArray[np.floating]:
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
        probsVec = probs(self.circuit.qnode(vec), 2)
        print(probsVec)
        centroids, norms2 = self.circuit.train()

        dists: list = list([])
        for i in range(norms2.shape[0]):
            Z: np.float64 = (norms2[i] + np.linalg.norm(vec) ** 2)
            dist: np.float64 = 4 * Z * (1 - probsVec[i]) 
            dists.append(dist)

        dists = np.array(dists)

        i: np.int64 = np.argmin(dists)
        return self.y[i]
    #end fit
#end QEuclidean

X_train = [
    [1.0, 1.0],
    [1.2, 0.8],
    [0.8, 1.1],
    [1.1, 1.3],
    [40.0, 40.0],
    [42.0, 38.0],
    [38.0, 41.0],
    [41.0, 43.0]
]
y_train = [0, 0, 0, 0, 1, 1, 1, 1]
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = [
    [1.1, 0.9],
    [39.9, 42.2],
    [25.0, 25.0]
]
y_test = [0, 1, '?']



df = pd.read_csv("/home/elma/Documentos/RAQun/instances.csv")
X = df.drop(columns=['Class/ASD'])
y = df['Class/ASD']

X = X.to_numpy()
y = y.to_numpy()

model = QEuclidean()
model.fit(X, y)

for i, vec in enumerate(X):
    pred = model.predict(vec)
    print(f"Vector: {vec}, Predicción: {pred}, Real: {y[i]}")