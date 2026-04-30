from RAQun.circuits import Circuit
from typing import Dict, Any, List
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray
from RAQun.utils.maths import log_t, probs
from RAQun.utils.qun import ctrlGen

class InnerProduct1R(Circuit):
    """
        Calculates the inner product of two quantum states by using Hadamard Test.

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
        
        self.logK = int(np.ceil(np.log2(self.vecs.shape[0])))
        self.n, self.logN = log_t(self.vecs.shape[0])
        self.d, self.logD = log_t(self.vecs.shape[1])
        
        self.P = ctrlGen(self.d, self.logD)
        self.Q = ctrlGen(self.vecs.shape[0], self.logK)
        
        self.dev = qml.device(
            "lightning.qubit", 
            wires=self.logK + self.logD + 2
        )
        self.qnode = qml.QNode(self.run, self.dev, shots=1024)
        
        self.regJ = [i for i in range(self.logK)]
        self.regI = [i for i in range(self.logK, self.logK + self.logD)]
        self.regV = self.logK + self.logD
        self.regA = self.logK + self.logD + 1
    #end __init__

    def run(self) -> Dict[str, Any] | Any:
        """
            Calculates the counts of the measurement of the circuit.

            Returns
            -------
            dict
                Counts of the measurement of the circuit.
        """

        for i in self.regJ:
            qml.Hadamard(wires=i)
        for i in self.regI:
            qml.Hadamard(wires=i)
        qml.Hadamard(wires=self.regA)
        
        for i, p in enumerate(self.P):
            theta = 2 * np.arcsin(np.clip(self.vec[i]/np.max(self.vecs), -1.0, 1.0))
            ctrl = p + '0'
            ctrl = [int(c) for c in ctrl]
            qml.ctrl(
                qml.RY(theta, wires=self.regV), 
                control=self.regI[:]+[self.regA], 
                control_values=ctrl
            )
            
        for j, q in enumerate(self.Q):
            for i, p in enumerate(self.P):
                theta = 2 * np.arcsin(np.clip(self.vecs[j][i]/np.max(self.vecs), -1.0, 1.0))
                ctrl = q + p + '1'
                ctrl = [int(c) for c in ctrl]
                qml.ctrl(
                    qml.RY(theta, wires=self.regV), 
                    control=self.regJ[:]+self.regI[:]+[self.regA], 
                    control_values=ctrl
                )

        qml.Hadamard(wires=self.regA)
        
        return qml.counts(wires=self.regJ[::]+[self.regA])
    #end run
#end InnerProduct1R

if __name__ == '__main__':
    vec = np.array([1, 2, 3])
    vecs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    circ = InnerProduct1R(vec, vecs)

    print(circ.qnode())