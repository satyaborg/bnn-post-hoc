import numpy as np
from src.vbutils.yeo_johnson import tau2eta, dtheta_dtau
from src.vbutils.grad_theta_logq_func import grad_theta_logq
from src.vbutils.dtheta_dmu_func import dtheta_dmu
from src.vbutils.dtheta_dBDelta_func import dtheta_dBDelta


def gradient_compute(theta, mu, B, z, d, eps, tau, logpost, phi, Transf="YJ"):
    q, p = B.shape
    eta = tau2eta(tau)
    g, delta_logh = logpost(theta)

    delta_logq = grad_theta_logq(theta, eta, mu, B, d, phi, Transf)
    L_mu = dtheta_dmu(phi, eta, theta, Transf) @ (delta_logh - delta_logq)
    dtheta_dB, dtheta_dd = dtheta_dBDelta(eta, B, z, eps, phi, theta, Transf)
    L_B = np.reshape(dtheta_dB.transpose() @ (delta_logh - delta_logq), (q, p), "F")
    L_d = dtheta_dd.transpose() @ (delta_logh - delta_logq)
    L_tau = (dtheta_dtau(phi, tau, theta, Transf).transpose()) @ (
        delta_logh - delta_logq
    )

    return L_mu, L_B, L_d, L_tau, g
