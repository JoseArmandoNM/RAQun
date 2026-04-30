from abc import ABC, abstractmethod
from typing import Any, Dict

class Algorithm(ABC):
    """
    Defines a common interface for algorithm-like objects
    """
    labels_: Any
    @abstractmethod
    def fit(self, X: Any) -> Any:
        """
            Fit the algorithm to the data.

            Parameters
            ----------
            X : NDArray[np.floating]
                Matrix of shape (nFeatures, nSamples) with the data.
        """
        raise NotImplementedError("Subclasses must implement this method")
    #end fit

    def fit_predict(self, X: Any) -> Any:
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
    #end fit_predict

    def get_params(self) -> Dict[str, Any]:
        """
            Get the parameters of the algorithm.

            Returns
            -------
            dict
                Dictionary with the parameters of the algorithm.
        """
        return self.__dict__
    #end get_params

    def set_params(self, **params: Any) -> 'Algorithm':
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
    #end set_params
#end Algorithm
