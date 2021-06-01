import sys
import numpy as np


def vec2mat(vec, m, n):
    if vec.shape[0] != (m * n):
        sys.exit("The length of vec must equal m*n")

    mat = np.reshape(vec, (m, n), "F")
    return mat


from scipy.sparse import csr_matrix


def communtationM(A):
    m, n = A.shape[0], A.shape[1]
    row = np.arange(m * n)
    col = row.reshape((m, n), order="F").ravel()
    data = np.ones(m * n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m * n, m * n))
    return K


def vec(A):
    m, n = A.shape[0], A.shape[1]
    return A.reshape(m * n, order="F")
