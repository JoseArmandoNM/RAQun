from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen

class InnerProduct(Circuit):
    """
        Calculates the inner product of two quantum states by using Swap Test.

        Parameters
        ----------
        vec : NDArray[np.floating]
            Vector of shape (nFeatures,).
        vecs : NDArray[np.floating]
            Matrix of shape (nSamples, nFeatures) with the data.
    """

    def __init__(self, vec: NDArray[np.floating], vecs: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector of shape (nFeatures,).
            vecs : NDArray[np.floating]
                Matrix of shape (nSamples, nFeatures) with the data.
        """
        self.vec = vec
        self.vecs = vecs
        
        
        self.log_c = int(np.ceil(np.log2(self.vecs.shape[0])))
        self.d, self.log_d = log_t(self.vecs.shape[1])
        
        self.Q = ctrlGen(self.vecs.shape[0], self.log_c)
        
        self.dev = qml.device(
            "lightning.qubit", 
            wires=self.log_c + 2 * self.log_d + 1
        )
        self.qnode = qml.QNode(self.run, self.dev, shots=10024)
        
        self.reg_J = [i for i in range(self.log_c)]
        self.reg_U = [i for i in range(self.log_c, self.log_c + self.log_d)]
        self.reg_V = [i for i in range(self.log_c + self.log_d, self.log_c + 2 * self.log_d)]
        self.reg_A = [i for i in range(self.log_c + 2 * self.log_d, self.log_c + 2 * self.log_d + 1)]
    #end __init__

    def normalize(self, vec: NDArray[np.floating]) -> NDArray[np.floating]:
        """
            Normalizes a vector.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector to normalize.

            Returns
            -------
            NDArray[np.floating]
                Normalized vector padded to 2**log_d size.
        """

        vec = np.pad(vec, (0, 2**self.log_d - vec.size), mode='constant')
        vec = vec / np.linalg.norm(vec)
        return vec
    #end normalize

    def initialize(self, C: NDArray[np.floating], wires: list) -> None:
        """
            Initializes the quantum states using amplitude encoding.

            Parameters
            ----------
            C : NDArray[np.floating]
                Vector to encode.
            wires : list
                List of wires to encode the state into.
        """

        qml.StatePrep(self.normalize(C), wires=wires)
    #end initialize

    def run(self) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """

        x: NDArray[np.float64] = self.normalize(self.vec)

        for i in self.reg_J:
            qml.Hadamard(wires=i)
        
        for i in self.reg_A:
            qml.Hadamard(wires=i)
        
        self.initialize(x, self.reg_U)
        
        for i, q in enumerate(self.Q):
            q = [int(c) for c in q]
            qml.ctrl(
                self.initialize,
                control=self.reg_J,
                control_values=q 
            )(self.vecs[i], wires=self.reg_V)

        for i in self.reg_U:
            qml.CSWAP(wires=[self.reg_A[0], i, i+self.log_d])
        
        for i in self.reg_A:
            qml.Hadamard(wires=i)
        
        return qml.counts(wires=self.reg_J[:] + self.reg_A[:])
    #end run
#end InnerProduct


if __name__ == '__main__':
    vec = np.array([1, 2, 3])
    vecs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    circ = InnerProduct(vec, vecs)

    print(circ.qnode())
