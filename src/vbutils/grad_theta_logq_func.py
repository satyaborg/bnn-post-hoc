import scipy.sparse as sp
import numpy as np
from src.vbutils.yeo_johnson import dtYJ_dtheta


def grad_theta_logq(theta, eta, mu_phi, B, Delta, phi, Transf_type="YJ"):
    n = theta.shape[0]
    K = B.shape[1]
    Tq1vec = np.zeros((n, 1))
    Dm2 = (sp.diags((np.power(Delta, -2)).transpose(), [0])).toarray()
    IK = np.eye(K)
    Bt = B.transpose()
    Sigma_phiinv = Dm2 - (Dm2 @ B) @ np.linalg.inv(IK + Bt @ Dm2 @ B) @ Bt @ Dm2

    dt_dtheta = dtYJ_dtheta(theta, eta)
    c2 = theta < 0
    c3 = 0 <= theta
    Tq1vec[c2] = (eta[c2] - 1) / (-theta[c2] + 1)
    Tq1vec[c3] = (eta[c3] - 1) / (theta[c3] + 1)
    dtdtheta = sp.diags(dt_dtheta.transpose(), [0])
    Tq2vec = -dtdtheta.transpose() @ Sigma_phiinv @ (phi - mu_phi)
    T = Tq1vec + Tq2vec

    return T
