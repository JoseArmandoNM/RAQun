from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def get_params(self):
        return self.__dict__

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self