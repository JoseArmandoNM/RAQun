from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

def log_2(num: int) -> int:
    """
        Calculates the ceiling of the base-2 logarithm of a number.

        Parameters
        ----------
        num : int
            Number to calculate the logarithm of.

        Returns
        -------
        int
            Ceiling of the base-2 logarithm of the number.
    """
    return int(np.ceil(np.log2(num))) if num > 1 else 1
#end log_2

def log_t(num: int) -> tuple[int, int]:
    """
        Calculates the base-2 logarithm of a number.

        Parameters
        ----------
        num : int
            Number to calculate the logarithm of.

        Returns
        -------
        int
            Ceiling of the base-2 logarithm of the number.
        int 
            Number given as a parameter
    """
    return num, int(np.ceil(np.log2(num))) if num > 1 else 1
#end log_t

def dist(probs: NDArray[np.floating], method: str = 'hadamard', norms2: NDArray[np.floating] = None) -> float:
    """
        Calculates the Euclidean distance between two vectors.

        Parameters
        ----------
        probs : NDArray[np.floating]
            Probabilities of each element.
        method : str
            Method to calculate the inner product.
            It only accepts 'hadamard' or 'swap'.
            Does not distinguish lower or upper case.
        norms2 : NDArray[np.floating]
            Vector of squared norms of the vectors.
            The first element is the squared norm of the vector to compare against.
            The rest of the elements are the squared norms of the vectors to compare with.

        Returns
        -------
        float
            Approximation of the euclidean distance between a vector and a list of vectors.
    """

    method = method.lower()
    if method not in ['hadamard', 'swap']:
        raise ValueError(f'Method {method} not recognized.')
    
    dists = np.array([], dtype=np.float64)
    vec = norms2[0]
    vecs = norms2[1:]
    for i in range(len(probs)):
        z = 1 if method == 'hadamard' else vec + vecs[i]
        dists = np.append(dists, 4*z*(1-probs[i]))

    return dists
#end dist

def probs(counts: dict[str, int], nStates: int) -> NDArray[np.floating]:
    """
        Calculates the probabilities of each element in the dictionary.

        Parameters
        ----------
        counts : dict[str, int]
            Dictionary of counts.
        nStates : int
            Number of states in superposition.

        Returns
        -------
        NDArray[np.floating]
            Probabilities of each element.
    """
    probs = np.array([], dtype=np.float64)
    
    counts_idx = {}
    counts_match = {}
    
    for bit_string, count in counts.items():
        idx_bits = bit_string[:-1]
        ancilla = bit_string[-1]
        
        idx_decimal = int(idx_bits, 2)
        
        counts_idx[idx_decimal] = counts_idx.get(idx_decimal, 0) + count
        if ancilla == '0':
            counts_match[idx_decimal] = counts_match.get(idx_decimal, 0) + count

    for i in range(nStates):
        total = counts_idx.get(i, 0)
        match = counts_match.get(i, 0)
        
        prob = match / total if total > 0 else 0.0
        probs = np.append(probs, prob)
        
    return probs
#end probs