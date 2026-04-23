from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen

class InnerProduct1R(Circuit):
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
        self.C, self.norms2 = self.train()
        self.log_k = np.ceil(np.log2(self.C.shape[0])).astype(int)
        self.n, self.log_n = log_t(X.shape[0])
        self.d, self.log_d = log_t(X.shape[1])
        self.P = ctrlGen(self.d, self.log_d)
        self.Q = ctrlGen(self.C.shape[0], self.log_k)
        self.dev = qml.device(
            "lightning.qubit", 
            wires=self.log_n + self.log_d + 2
        )
        self.qnode = qml.QNode(self.run, self.dev, shots=1024)
        self.reg_J = [i for i in range(self.log_k)]
        self.reg_I = [i for i in range(self.log_k, self.log_k + self.log_d)]
        self.reg_V = self.log_k + self.log_d
        self.reg_A = self.log_k + self.log_d + 1
    #end __init__

    def run(self, vec: NDArray[np.floating]) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Parameters
            ----------
            vec : NDArray[np.floating]
                Vector of shape (nFeatures,).

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """
        nn: NDArray[np.float64] = np.array([])

        for i in self.reg_J:
            qml.Hadamard(wires=i)
        for i in self.reg_I:
            qml.Hadamard(wires=i)
        qml.Hadamard(wires=self.reg_A)
        
        for i, p in enumerate(self.P):
            theta = 2 * np.arcsin(vec[i]/np.max(self.X))
            ctrl = p + '0'
            ctrl = [int(c) for c in ctrl]
            qml.ctrl(
                qml.RY(theta, wires=self.reg_V), 
                control=self.reg_I[:]+[self.reg_A], 
                control_values=ctrl
            )
            
        for j, q in enumerate(self.Q):
            for i, p in enumerate(self.P):
                theta = 2 * np.arcsin(self.C[j][i]/np.max(self.X))
                ctrl = q + p + '1'
                ctrl = [int(c) for c in ctrl]
                qml.ctrl(
                    qml.RY(theta, wires=self.reg_V), 
                    control=self.reg_J[:]+self.reg_I[:]+[self.reg_A], 
                    control_values=ctrl
                )

        qml.Hadamard(wires=self.reg_A)
        
        return qml.counts(wires=self.reg_J[::]+[self.reg_A])
    #end run

    def train(self) -> NDArray[np.float64]: 
        centroids = np.array([ self.X[self.labels==c].mean(axis=0) for c in self.Y], dtype=np.float64)
        normas = np.array([np.linalg.norm(c) ** 2 for c in centroids], dtype=np.float64)
        
        return centroids, normas
    #end train


#end InnerProduct1R