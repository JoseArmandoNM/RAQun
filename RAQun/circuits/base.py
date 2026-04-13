from abc import ABC, abstractmethod

class Circuit(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)