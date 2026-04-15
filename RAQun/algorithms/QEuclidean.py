from RAQun.algorithms.base import Algorithm

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
        pass

    def train(self) -> NDArray[np.floating]:
        """
        Calculates the mean of each class.

        Returns
        -------
        NDArray[np.floating]
            Array of shape (nClasses, nFeatures) with the mean of each class.
        """
        pass

    def fit(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
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
        