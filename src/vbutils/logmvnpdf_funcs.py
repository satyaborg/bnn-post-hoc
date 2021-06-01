import numpy as np
import scipy.sparse as sp


def logmvnpdf(x, mu, Sigma, CholSigma=None):
    if CholSigma is None:
        CholSigma = np.linalg.cholesky(Sigma)

    # outputs log likelihood array for observations x  where x_n ~ N(mu,Sigma)
    # x is NxD, mu is 1xD, Sigma is DxD

    Dims = x.shape
    D = Dims[1]
    const = -0.5 * D * np.log(2 * np.pi)

    xc = x - mu
    Sigma_inv = np.linalg.inv(Sigma)

    term1 = -0.5 * (np.multiply(np.matmul(xc, Sigma_inv), xc).sum(axis=1))
    term2 = const - 0.5 * logdet2(Sigma, CholSigma)
    logp = term1.transpose() + term2
    return logp


def logdet2(A, U=None):
    if U is None:
        U = np.linalg.cholesky(A)

    y = 2 * sum(np.log(abs(np.diag(U))))
    return y


def logmvnpdf2(B, z, d, eps):
    p, q = B.shape
    Bt = B.transpose()
    Bz_deps = B @ z + d * eps
    DBz_deps = Bz_deps * np.power(d, -2)
    Dinv2B = B * np.power(d, -2)
    Iq = sp.csr_matrix(np.eye(q))

    Half1 = DBz_deps
    Half2 = Dinv2B @ np.linalg.inv(Iq + Bt @ Dinv2B) @ Bt @ DBz_deps

    Blogdet = np.log(np.linalg.det(Iq + Dinv2B.transpose() @ B)) + sum(
        np.log(np.power(d, 2))
    )
    logp = -(
        p / 2 * np.log(2 * np.pi)
        + 1 / 2 * Blogdet
        + 1 / 2 * Bz_deps.transpose() @ (Half1 - Half2)
    )

    return logp
