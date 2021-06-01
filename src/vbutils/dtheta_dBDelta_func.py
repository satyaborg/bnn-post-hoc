import numpy as np
import scipy.sparse as sp
from src.vbutils.yeo_johnson import dtheta_dphi


def dtheta_dBDelta(eta, B, z, epsilon, phi, theta, Transf_type="YJ"):
    n = B.shape[0]
    TB1 = sp.csr_matrix(np.kron(z.transpose(), np.eye(n)))
    TD1 = sp.diags(epsilon.transpose(), [0])

    dthetdphi = dtheta_dphi(phi, eta)
    dthetadphi = sp.diags(dthetdphi.transpose(), [0])
    dthetadB = dthetadphi @ TB1
    dthetadDelta = dthetadphi @ TD1
    return dthetadB, dthetadDelta
