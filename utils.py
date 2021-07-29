import numpy as np
import BinOps

def xorSum(indexArray,dataArray):
    """
    accept a array of index and data array(numpy)\n
    do successive xor of npArray[index]
    """
    x = np.zeros(dataArray.shape[1])
    for index in indexArray:
        x = np.logical_xor(x, dataArray[index])
    return x


def weight(x):
    """
    calculate hamming weight
    """
    return np.count_nonzero(x)


def generatorMatrix(n, k, generator_poly):
    """
    input: code length\n
    return: Systematic Generator Matrix and Check Matrix(http://www.rutvijjoshi.co.in/index_files/lecture-26.pdf)
    """
    p = np.zeros((k,n-k))
    for i in range(0,k):
        remainder = np.polydiv(np.eye(n)[i], generator_poly)[1]
        p[i][k-remainder.shape[0]-1:] = remainder
    p = np.remainder(p, 2)
    generatorMatrix = np.block([np.eye(k), p])
    checkMatrix = np.block([p.T, np.eye(n-k)])
    return generatorMatrix, checkMatrix


def errorTable(n, H):
    """
    input: n(code length), H(check matrix)\n
    return: emTable(error table), smTable(syndrome table)
    """
    x, n = H.shape # H.shape is (n-k,n)
    k = n-x
    emTable = np.concatenate((np.identity(k), np.zeros((k,x)) ), axis=1)
    smTable = np.mod(np.matmul(emTable, H.T), 2) #i.e the syndrome table T
    return emTable, smTable


def polyDiv(dividend, divisor):
    """
    Do polynomial long division in GF(2)
    :param dividend: numpy array
    :param divisor: numpy array
    :return: [quotient, remainder]
    """
    a = dividend[::-1]
    b = divisor[::-1]
    quo, rem = BinOps.BinPolyDiv(list(a), list(b))
    quoRev = quo[::-1]
    remRev = rem[::-1]
    return np.asarray(quoRev), np.asarray(remRev)


def nonSysGenMatrix(n, k, genPoly):
    """
    input: code length ,origin bit length, and generator polynomial\n
    genPoly coefficients from x^n-k to x^0\n
    return: NON Systematic Generator Matrix and Check Matrix
    """
    genPolyPadded = np.block([genPoly, np.zeros((1, n-genPoly.shape[0]))])
    G = np.zeros((k, n))
    for i in range(0, k):
        G[i, :] = np.roll(genPolyPadded, i)
    # h(x) = x^n + 1 / g(x)
    product = np.zeros(n + 1)
    product[0] = 1
    product[-1] = 1
    hx = polyDiv(product, genPoly)[0]
    hxReverse = hx[::-1]
    H = np.zeros((n-k, n))
    hxPadded = np.block([np.zeros((1, n-hx.shape[0])), hxReverse])
    for i in range(0, n-k):
        H[i, :] = np.roll(hxPadded, -i)
    return G, H


def is_codeword(c, H):
    tmp = c @ H.T
    tmp = np.remainder(tmp, 2)
    tmp = np.sum(tmp)
    return not tmp


def saveMatrix(filename: str, matrix):
    """
    :param filename: Path of matrix to be saved
    :param matrix: NumPy matrix
    :return:
    """
    with open(filename, "w") as f:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                f.write(str(int(matrix[i, j])))
                f.write(' ')
            f.write('\n')


def saveMat_OL_Li(filename: str, mat):
    """
    Save ajacency list
    1st row: number of columns of matrix
    2nd row: number of rows of matrix
    3rd row: max row weight
    4th row: total weight of matrix
    each row below corresponds to each row of thee original matrix:
    numbers in a row correspond to the indice(grow from 1) of element '1' in matrix 
    if the numbers less than max row weight, fill with 0

    :param filename: Path of matrix to be saved
    :param matrix: NumPy matrix
    """
    with open(filename, "x") as f:
        f.write(str(mat.shape[1]))  # num of cols
        f.write("\n")
        f.write(str(mat.shape[0]))  # num of rows
        f.write("\n")
        rowWeight = np.sum(mat, axis=1)
        maxRowWeight = int(np.max(rowWeight))
        f.write(str(maxRowWeight))
        f.write("\n")
        totalWeight = int(np.sum(mat))
        f.write(str(totalWeight))
        f.write("\n")
        for i in range(mat.shape[0]):
            count = 0
            for j in range(mat.shape[1]):
                if mat[i, j] == 1:
                    f.write(str(j+1))
                    f.write(" ")
                    count += 1
            if count < maxRowWeight:
                for k in range(maxRowWeight - count):
                    f.write("0")
            f.write("\n")
