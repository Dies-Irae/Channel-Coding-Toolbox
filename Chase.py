import numpy as np
from itertools import combinations

import utils


class Chase:
    def __init__(self, code, algebraic_method):
        """
        Chase Decoding Algorithm
        :param code: An object of Linear Code class
        :param algebraic_method: Function reference of an algebraic decoding algorithm which can correct up to d errors.
        """

        self.codeLen = code.n
        self.T_size = 2**(code.d//2)
        self.T = np.zeros((self.T_size, code.n))
        self.d_min = code.d
        self.alg_dec = algebraic_method
        self.H = code.H

    def generate_T(self, codeword):
        """
        Using Variant 2: T is set of all binary vectors combinations of t' = floor(d_min/2) least reliable vectors,
        in this case, |T| = 2^t
        """
        self.T = np.zeros_like(self.T)
        unreliable_position = np.argpartition(np.abs(codeword), self.d_min // 2)[: self.d_min // 2]
        j = 0
        for k in range(self.d_min // 2 + 1):
            combs = combinations(unreliable_position, k)
            for reverse_bits in combs:
                self.T[j][list(reverse_bits)] = 1
                j += 1

    def decode(self, r):
        self.generate_T(r)
        y = r.copy()
        y[y > 0] = 0
        y[y < 0] = 1
        Y = np.remainder(self.T+y, 2)
        L = []
        for i in range(self.T_size):
            Y[i] = self.alg_dec(Y[i])
            if utils.is_codeword(Y[i], self.H):
                L.append(i)
        maxDist = 0
        for i in L:
            tmpDist = np.sum((1-2*Y[i])*r)
            if tmpDist > maxDist:
                maxDist = tmpDist
                maxIndex = i
        if maxDist == 0:
            return y
        else:
            return Y[maxIndex]
