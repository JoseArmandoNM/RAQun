from abc import ABC, abstractmethod

class Circuit(ABC):
    """
    Defines a common interface for circuit-like objects
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Execute the circuit with the given inputs.
        This method must be implemented by subclasses to define the circuit's execution logic.

        Parameters
        ----------
        *args : Any
            Positional arguments required for execution.
        **kwargs : Any
            Keyword arguments required for execution.
            
        Returns
        -------
        Any
            The result of the circuit execution.
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Call the circuit as a function.
        This method allows instances of the class to be invoked directly, forwarding all arguments to the `run` method.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to `run`.
        **kwargs : Any
            Keyword arguments passed to `run`.

        Returns
        -------
        Any
            The result of the circuit execution.
        """
        return self.run(*args, **kwargs)