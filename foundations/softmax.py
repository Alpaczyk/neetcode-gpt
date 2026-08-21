import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res = np.array([]).astype('float64')
        for i in z:
            nominator = np.e ** (i - max(z))
            denominator = sum(np.e ** (j - max(z)) for j in z)
            res = np.append(res, round((nominator / denominator), 4))
        return res