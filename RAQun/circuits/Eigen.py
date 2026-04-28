from RAQun.circuits import Circuit
from typing import Dict
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen, padder

class Eigen(Circuit):
    """
        Calculates the eigenvalues and eigenvectors of a matrix using the Quantum Phase Estimation algorithm.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nFeatures, nSamples) with the data.
    """

    def __init__(self, X: NDArray[np.floating]) -> None:
        """
            Initializes the class variables.

            Parameters
            ----------
            X : NDArray[np.floating]
                Matrix of shape (nFeatures, nSamples) with the data.
        """

        self.X = (X + X.T)/2
        self.norm = np.linalg.norm(X, ord=2)
        self.X = self.X / self.norm
        self.X = padder(self.X)
        self.X = self.X + 0.5 * np.eye(self.X.shape[0])
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if self.X[i][j] == 0:
                    self.X[i][j] = 1e-10
        self.shots = 1024
        self.n, self.log_n = log_t(X.shape[0])
        self.m, self.log_m = log_t(500)
        self.Q = ctrlGen(self.n, self.log_n)
        self.reg_A = list(range(self.log_m))
        self.reg_I = list(range(self.log_m, self.log_m + self.log_n))
        self.dev = qml.device(
            "default.qubit", 
            wires = len(self.reg_A) + len(self.reg_I)
        )
        self.qnode = qml.QNode(
            self.run, 
            self.dev, 
            shots=self.shots
        )
        self.max = np.max(self.X)
    #end __init__

    def run(self) -> dict:
        """
            Calculates the counts of the measurement of the circuit.

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """

        for i in self.reg_A:
            qml.Hadamard(wires = i)
        
        for j, i in enumerate(self.reg_I):
            qml.RY(0.1 + 0.2 * j, wires=i)

        for i in self.reg_I:
            qml.Hadamard(wires = i)

        for i in range(len(self.reg_A)):
            t_evol = 2 ** i
            qml.ctrl(
                self.oracle, 
                control = self.reg_A[i]
            )(
                t_evol, 
                wires = self.reg_I[:]
            )

        qml.adjoint(qml.QFT)(wires = self.reg_A[:])
        return qml.counts(wires=self.reg_A[:]+self.reg_I[:])
    #end run

    def oracle(self, t: int, wires: list) -> Any:
        """
            Oracle that implements the unitary operator U.

            Returns
            -------
            qml.QubitUnitary
                Expected values of the oracle that implements the unitary operator U.
        """
        H = qml.Hermitian(self.X, wires = wires)
        return qml.exp(H, -1j*t)
    #end oracle

    def vectors(self, k: int) -> NDArray[np.floating]:
        """
            Calculates the eigenvectors of the matrix.

            Parameters
            ----------
            k : int
                Number of eigenvectors to calculate.

            Returns
            -------
            NDArray[np.floating]
                Matrix of shape (nFeatures, k) with the eigenvectors.
        """

        p = list(self.probs().values())
        mat = []

        for _ in range(2*k):
            kAux = _*2**self.log_n
            pAux = p[kAux: kAux + self.n]
            mat.append(pAux)
        
        vecs = np.array(mat)

        return np.transpose(vecs)
    #end vectors

    def probs(self) -> dict:
        probs = self.qnode()
        probsAux = {}
        k = self.log_m + self.log_n
        states = ctrlGen(2**k, k)
        for s in states:
            if s in probs.keys():
                probsAux[s] = probs[s]
            else:
                probsAux[s] = 0
        return probsAux
    #end probs
#end Eigen
