import numpy as np
import math

class Simulation:
    def __init__(self, n, k, G, H):
        """
        :param n: code length, integer
        :param k: bit length before encoding, integer
        :param G: k*n generate matrix, numpy array
        :param H: (n-k)*n parity check matrix, numpy array
        """
        self.n = n
        self.k = k
        self.G = G
        self.H = H

    def encode(self, message):
        """
        Input: origin message\n
        Return: encoded codeword
        """
        return np.remainder(np.matmul(message, self.G), 2)

    def generateBatch(self, BatchSize):
        """
        return a randomly generated message in NumPy array form, which size is(infomation bits length(k), BatchSize)
        """
        batch = np.random.randint(2, size=(BatchSize, self.k))
        return batch

    def AWGN(self, inputs, SNR):
        """
        inputs: codewords to be send\n
        SNR: Eb/N0 signal noise ratio(dB)
        """
        inputs = -(inputs * 2 - 1)  # BPSK modulation
        sigma = math.sqrt(1 / (2 * self.k / self.n * math.pow(10, SNR / 10)))
        inputs = inputs + np.random.normal(0, sigma, inputs.shape)
        return inputs
