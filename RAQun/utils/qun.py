from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

def ctrlGen(d: int, logd: int) -> NDArray[np.str_]:
    """
        Generates a vector of control strings for the Q-UN algorithm.

        Parameters
        ----------
        d : int
            Number of control strings.
        logd : int
            Logarithm of the number of control strings.

        Returns
        -------
        NDArray[np.str_]
            Vector of control strings.
    """
    P = []
    numsD = set([i for i in range(logd)])
    for i in range(d):
        B = [int(p) for p in bin(i)[2:]]
        append = int(logd - len(B))
        for _ in range(append): 
            B.insert(0, 0)

        p = ''.join(str(_) for _ in B)
        P.append(p)

    return P
#end ctrlGen

def padder(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
        Pads the given matrix with zeros to make it a square (2^k, 2^K) matrix.

        Parameters
        ----------
        X : NDArray[np.floating]
            Matrix of shape (nSamples, nFeatures) with the data.

        Returns
        -------
        NDArray[np.floating]
            Matrix of shape (2^k, 2^K) with the data padded with zeros.
    """

    n = X.shape[0]
    log_n = int(np.ceil(np.log2(n)))
    required = 2**log_n
    diff = int(required - n)
    
    return np.pad(X, ((0, diff), (0, diff)), 'constant')
#end padder
