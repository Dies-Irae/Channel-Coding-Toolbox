from itertools import combinations
from utils import *


class QRCode:
    def __init__(self, n):
        self.n = n
        if n == 47:
            self.k = 24
            self.d = 11
            self.t = 5
            self.generatorPolynomial = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])

        elif n == 71:
            self.k = 36
            self.d = 11
            self.t = 5
            self.generatorPolynomial = np.array(
                [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
                 1, 1])

        elif n == 73:
            self.k = 37
            self.d = 13
            self.t = 6
            self.generatorPolynomial = np.array(
                [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                 0, 1, 1])

        elif n == 89:
            self.k = 45
            self.d = 17
            self.t = 8
            self.generatorPolynomial = np.array(
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
                 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])

        elif n == 97:
            self.k = 49
            self.d = 15
            self.t = 7
            self.generatorPolynomial = np.array(
                [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])

        elif n == 113:
            self.k = 57
            self.d = 15
            self.t = 7
            self.generatorPolynomial = np.array(
                [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])

        elif n == 127:
            self.k = 64
            self.d = 19
            self.t = 9
            self.generatorPolynomial = np.array(
                [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
                 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1])

        self.G, self.H = generatorMatrix(n, self.k, self.generatorPolynomial)
        self.emTable, self.smTable = errorTable(self.n, self.H)

    def _DSAlgorithm(self, r):
        """
        Decode the received bit vectors using DS algorithm(from "Using the Difference of Syndromes to Decode
        Quadratic Residue Codes Fig.1")\n inputs: the received bit vectors in (,k) size
        """
        r = r.copy()
        tau = 0  # step 1)
        s = np.remainder(np.matmul(r, self.H.T), 2)  # step 2)

        while weight(s) > self.t:
            # step 4)
            for we in range(1, self.t // 2 + 1):
                invertCombinations = combinations(range(self.k), we)
                for eta in invertCombinations:
                    eta = list(eta)
                    sd = np.logical_xor(s, xorSum(eta, self.smTable))  # invert one group info bits
                    wsd = weight(sd)
                    if wsd <= self.t - we:
                        dc = r.copy()
                        dc[eta] = 1 - dc[eta]
                        dc = np.logical_xor(dc, np.concatenate((np.zeros(self.k), sd)))
                        if tau != 0:  # step 7)
                            dc = np.roll(dc, self.k)
                        return dc  # step 8)
            tau += 1
            if tau == 1:
                r = np.roll(r, -self.k)  # step 5)
            elif tau == 2:
                r[-self.k] = 1 - r[-self.k]  # r = r +(1 << (k-1))n  # step 6)
            else:
                r[-self.k] = 1 - r[-self.k]
                r = np.roll(r, self.k)
                return r
            s = np.remainder(np.matmul(r, self.H.T), 2)  # step 2)

        dc = np.logical_xor(r, np.concatenate((np.zeros(self.k), s)))  # step 3)

        if tau != 0:
            dc = np.roll(dc, self.k)  # step 7)
        return dc  # step 8)

    def DSDecode(self, inputs):
        inputs[inputs > 0] = 0  # step 2
        inputs[inputs < 0] = 1
        decoded = np.zeros((inputs.shape[0], self.n))
        for i in range(inputs.shape[0]):
            decoded[i] = self._DSAlgorithm(inputs[i])
        return decoded

