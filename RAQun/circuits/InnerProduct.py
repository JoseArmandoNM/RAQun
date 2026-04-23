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
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.
        y : NDArray[np.floating]
            Vector of shape (nFeatures,) with the label.
    """

    def __init__(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            X : NDArray[np.floating]
                Matrix of shape (nFeatures, nSamples) with the data.
            y : NDArray[np.floating]
                Vector of shape (nFeatures,) with the label.
        """
        self.X = X
        self.labels = y
        self.Y = np.unique(self.labels)
        self.nQubits: int = self.nQubitsCalc()
        self.C, self.norms2 = self.train()
        self.log_k = np.ceil(np.log2(self.C.shape[0])).astype(int)
        self.n, self.log_n = log_t(self.X.shape[0])
        self.d, self.log_d = log_t(self.C.shape[1])
        self.c, self.log_c = log_t(self.C.shape[0])
        self.P = ctrlGen(self.d, self.log_d)
        self.Q = ctrlGen(self.n, self.log_n)
        self.dev = qml.device(
            "lightning.qubit", 
            wires=self.log_c + 2*self.log_d + self.log_k + 1
        )
        self.qnode = qml.QNode(self.run, self.dev, shots=10024)
        self.reg_J = [i for i in range(self.log_c)]
        self.reg_U = [i for i in range(self.log_c, self.log_c + self.log_d)]
        self.reg_V = [i for i in range(self.log_c + self.log_d, self.log_c + 2*self.log_d)]
        self.reg_A = [i for i in range(self.log_c + 2*self.log_d, self.log_c + 2*self.log_d + 1)]
    #end __init__

    def nQubitsCalc(self) -> int:
        """
            Calculates the number of qubits required for the circuit.

            Returns
            -------
            int
                Number of qubits required by the circuit.
        """
        return int(np.ceil(np.log2(self.X[0].size)))
    #end nQubitsCalc

    def train(self) -> NDArray[np.float64]: 
        centroids = np.array([
            self.X[self.labels==c].mean(axis=0) for c in self.Y], 
            dtype=np.float64
        )
        normas = np.array([
            np.linalg.norm(c) ** 2 for c in centroids], 
            dtype=np.float64
        )

        return centroids, normas
    #end train

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
                Normalized vector.
        """
        vec = np.pad(vec, (0, 2**self.nQubits - vec.size), mode='constant')
        vec = vec / np.linalg.norm(vec)
        vec = vec / np.sqrt(np.sum(np.abs(vec)**2))
        return vec
    #end normalize

    def initialize(self, C: NDArray[np.floating]) -> None:
        """
            Initializes the quantum states.

            Parameters
            ----------
            C : NDArray[np.floating]
                Vector of shape (nFeatures,) with the label.
        """
        qml.StatePrep(self.normalize(X), wires=wires)
    #end initialize

    def run(self, vec: NDArray[np.floating]) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector of shape (nFeatures).

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """
        x: NDArray[np.float64] = self.normalize(x)
        probs: NDArray[np.float64] = np.array([])

        for i in self.reg_J:
            qml.Hadamard(wires=i)
        
        for i in self.reg_A:
            qml.Hadamard(wires=i)
        
        Q = control_generator(self.C.shape[0], self.log_c)

        self.initialize(x, self.reg_U)
        for i, q in enumerate(Q):
            q = [int(c) for c in q]
            qml.ctrl(
                self.initialize,
                control=self.reg_J,
                control_values = q 
            )(self.C[i], wires=self.reg_V)

        for i in self.reg_U:
            qml.CSWAP(wires=[self.reg_A[0], i, i+self.log_d])

        
        for i in self.reg_A:
            qml.Hadamard(wires=i)
        
        return qml.counts(wires=self.reg_J[:] + self.reg_A[:])
    #end run
#end InnerProduct
