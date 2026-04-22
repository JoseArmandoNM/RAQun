from RAQun.algorithms.base import Algorithm
from RAQun.circuits import InnerProduct, InnerProduct1R

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

    def __init__(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            X : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).
            y : NDArray[np.floating]
                Labels of shape (nSamples).
        """
        self.X = X
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
        pass
    #end train

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
            Predicts the class of each sample in the data matrix.

            Parameters
            ----------
            X : NDArray[np.floating]
                Data matrix of shape (nSamples, nFeatures).

            Returns
            -------
            NDArray[np.floating]
                Array of shape (nSamples,) with predicted class for each sample.
        """
        pass
    #end fit
