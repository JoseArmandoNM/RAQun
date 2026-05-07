from raqun.algorithms import QDBSCAN
import numpy as np
qdbscan = QDBSCAN(eps=1, minSamples=2)
labels = qdbscan.fit(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]))
print(labels)
