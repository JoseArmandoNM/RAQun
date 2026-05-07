from raqun.circuits import Circuit
from typing import Dict, List, Any
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from raqun.utils.maths import log_t, probs
from raqun.utils.qun import ctrlGen

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
        
        
        self.logC = int(np.ceil(np.log2(self.vecs.shape[0])))
        self.d, self.logD = log_t(self.vecs.shape[1])
        
        self.Q = ctrlGen(self.vecs.shape[0], self.logC)
        
        self.dev = qml.device(
            "lightning.qubit", 
            wires=self.logC + 2 * self.logD + 1
        )
        self.qnode = qml.QNode(self.run, self.dev, shots=10024)
        
        self.regJ = [i for i in range(self.logC)]
        self.regU = [i for i in range(
            self.logC, 
            self.logC + self.logD
        )]
        self.regV = [i for i in range(
            self.logC + self.logD, 
            self.logC + 2 * self.logD
        )]
        self.regA = [i for i in range(
            self.logC + 2 * self.logD, 
            self.logC + 2 * self.logD + 1
        )]
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
                Normalized vector padded to 2**logD size.
        """

        vec = np.pad(vec, (0, 2**self.logD - vec.size), mode='constant')
        vec = vec / np.linalg.norm(vec)
        return vec
    #end normalize

    def initialize(self, C: NDArray[np.floating], wires: List[int]) -> None:
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

    def run(self) -> Any:
        """
            Calculates the counts of the measurement of the circuit.

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """

        x: NDArray[np.float64] = self.normalize(self.vec)

        for i in self.regJ:
            qml.Hadamard(wires=i)
        
        for i in self.regA:
            qml.Hadamard(wires=i)
        
        self.initialize(x, self.regU)
        
        for i, q in enumerate(self.Q):
            q = [int(c) for c in q]
            qml.ctrl(
                self.initialize,
                control=self.regJ,
                control_values=q
            )(self.vecs[i], wires=self.regV)

        for i in self.regU:
            qml.CSWAP(wires=[self.regA[0], i, i+self.logD])
        
        for i in self.regA:
            qml.Hadamard(wires=i)
        
        return qml.counts(wires=self.regJ[:] + self.regA[:])
    #end run
#end InnerProduct


if __name__ == '__main__':
    vec = np.array([1, 2, 3])
    vecs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    circ = InnerProduct(vec, vecs)

    print(circ.qnode())
