import sys
import numpy as np
import scipy.sparse as sp


def tYJ(theta, eta):
    if theta.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    c2 = theta < 0
    c3 = 0 <= theta

    phi = np.zeros(theta.shape)
    phic2 = np.divide(-(np.power(-theta[c2] + 1, (2 - eta[c2])) - 1), 2 - eta[c2])
    phic3 = np.divide(np.power(theta[c3] + 1, eta[c3]) - 1, eta[c3])

    eta0c3 = eta == 0 & c3
    eta2c2 = eta == 2 & c2

    phi[c2] = phic2
    phi[c3] = phic3

    phi[eta0c3] = np.log(theta[eta0c3] + 1)
    phi[eta2c2] = -np.log(-theta[eta2c2] + 1)

    return phi


def tYJi(phi, eta):
    if phi.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    c2 = phi < 0
    c3 = 0 <= phi

    theta = np.zeros(phi.shape)
    thetac2 = 1 - np.power(
        np.multiply(1 - phi[c2], 2 - eta[c2]), np.power(2 - eta[c2], -1)
    )
    thetac3 = np.power(1 + np.multiply(phi[c3], eta[c3]), np.power(eta[c3], -1)) - 1

    eta0c3 = eta == 0 & c3
    eta2c2 = eta == 2 & c2
    theta[c2] = thetac2
    theta[c3] = thetac3
    theta[eta0c3] = np.exp(phi[eta0c3]) - 1
    theta[eta2c2] = 1 - np.exp(-phi[eta2c2])

    return theta


def eta2tau(eta, Transf_type="YJ"):
    eta_max = 2
    eta_min = 0
    tau = -np.log(((eta_max - eta_min) / (eta - eta_min)) - 1)
    return tau


def tau2eta(tau, Transf_type="YJ"):
    eta_max = 2
    eta_min = 0
    eta = (eta_max - eta_min) / (np.exp(-tau) + 1) + eta_min
    eta[eta == 2] = 1.999
    eta[eta == 0] = 0.001
    return eta


def deta_dtau(tau, Transf_type="YJ"):
    eta_max = 2
    eta_min = 0
    detadtau = (eta_max - eta_min) * np.exp(-tau) / np.power((np.exp(-tau) + 1), 2)
    return detadtau


def dtheta_dphi(phi, eta):
    if phi.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    eta[abs(eta) < 0.0000001] = 0.0000001
    c2 = phi < 0
    c3 = 0 <= phi
    dthetadphi = np.zeros(phi.shape)
    dthetadphi[c2] = np.power(
        1 - np.multiply(phi[c2], 2 - eta[c2]), (eta[c2] - 1) / (2 - eta[c2])
    )
    dthetadphi[c3] = np.power(
        1 + np.multiply(phi[c3], eta[c3]), (1 - eta[c3]) / eta[c3]
    )
    return dthetadphi


def dtYJ_dtheta(theta, eta):
    if theta.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    eta[abs(eta) < 0.0000001] = 0.0000001
    c2 = theta < 0
    c3 = 0 <= theta
    dphi_dtheta = np.zeros(theta.shape)
    dphi_dtheta[c2] = np.power(-theta[c2] + 1, 1 - eta[c2])
    dphi_dtheta[c3] = np.power(theta[c3] + 1, eta[c3] - 1)
    return dphi_dtheta


def ddtYJ_dtheta(theta, eta):
    dt_dtheta = dtYJ_dtheta(theta, eta)
    c2 = theta < 0
    c3 = 0 <= theta
    Tq1vec = np.zeros(dt_dtheta.shape)

    Tq1vec[c2] = (eta[c2] - 1) / (-theta[c2] + 1)
    Tq1vec[c3] = (eta[c3] - 1) / (theta[c3] + 1)

    ddt_dtheta = Tq1vec * dt_dtheta
    return ddt_dtheta


def ddtYJ_dthetadeta(theta, eta):
    if theta.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    eta[abs(eta) < 0.0000001] = 0.0000001

    c2 = theta < 0
    c3 = 0 <= theta
    ddphi_dthetadeta = np.zeros(theta.shape)

    ddphi_dthetadeta[c2] = -np.power(-theta[c2] + 1, 1 - eta[c2]) * np.log(
        abs(-theta[c2] + 1)
    )
    ddphi_dthetadeta[c3] = np.power(theta[c3] + 1, eta[c3] - 1) * np.log(theta[c3] + 1)

    return ddphi_dthetadeta


def dtheta_deta(phi, eta):
    if phi.shape != eta.shape:
        sys.exit(
            "The number of inputs should be equal to the number of tranformation parameters"
        )

    eta[abs(eta) < 0.0000001] = 0.0000001

    c2 = phi < 0
    c3 = 0 <= phi
    dthetadeta = np.zeros(phi.shape)

    dthetadeta[c2] = -(
        np.power(1 - phi[c2] * (2 - eta[c2]), (1.0 / (2 - eta[c2])))
        * (
            phi[c2] / ((2 - eta[c2]) * (1 - phi[c2] * (2 - eta[c2])))
            + ((np.log(1 - phi[c2] * (2 - eta[c2]))) / (np.power(2 - eta[c2], 2)))
        )
    )
    dthetadeta[c3] = np.power(1 + phi[c3] * eta[c3], (1 / eta[c3])) * (
        (phi[c3] / (eta[c3] * (1 + phi[c3] * eta[c3])))
        - ((np.log(1 + phi[c3] * eta[c3])) / (np.power(eta[c3], 2)))
    )

    return dthetadeta


def dtheta_dtau(phi, tau, theta, Transf_type="YJ"):
    eta = tau2eta(tau)
    dthetadeta = dtheta_deta(phi, eta)
    detadtau = deta_dtau(tau)
    dthetadtau = sp.diags((dthetadeta * detadtau).transpose(), [0])

    return dthetadtau
