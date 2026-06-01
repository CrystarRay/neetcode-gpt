import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = z - np.max(z)
        exps = np.exp(shifted)
        softmax = exps / np.sum(exps)
        return np.round(softmax, 4)
