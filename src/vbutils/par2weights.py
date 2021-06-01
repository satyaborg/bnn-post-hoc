import numpy as np


def par2weights(theta):
    npar = np.prod(theta.shape)
    NNpar = int((npar - 2) / 3)
    tau = theta[0:NNpar]
    log_chi = theta[(NNpar + 1) - 1 : (2 * NNpar)]
    log_nu = theta[(2 * NNpar + 1) - 1 : (3 * NNpar)]
    log_xi = theta[3 * NNpar + 1 - 1]
    log_kappa = theta[3 * NNpar + 2 - 1]

    eta = np.sqrt(np.exp(log_chi)) * np.sqrt(np.exp(log_xi)) * tau

    return eta, log_chi, log_nu, log_xi, log_kappa, tau
