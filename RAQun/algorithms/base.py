from abc import ABC, abstractmethod

class Algorithm(ABC):
    """
    Defines a common interface for algorithm-like objects
    """

    @abstractmethod
    def fit(self, X):
        """
        Fit the algorithm to the data.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.
        """
        pass

    def fit_predict(self, X):
        """
        Fit the algorithm to the data and return the labels.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.

        Returns
        -------
        NDArray[np.floating]
            Vector of shape (nFeatures,) with the labels.
        """
        self.fit(X)
        return self.labels_

    def get_params(self):
        """
        Get the parameters of the algorithm.

        Returns
        -------
        dict
            Dictionary with the parameters of the algorithm.
        """
        return self.__dict__

    def set_params(self, **params):
        """
        Set the parameters of the algorithm.

        Parameters
        ----------
        **params : dict
            Dictionary with the parameters of the algorithm.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self