import numpy as np
from src.vbutils.igamma_logpdf import igamma_logpdf


def log_pri_log_phi(phi, log_phi, nu):
    log_p_log_phi = sum(igamma_logpdf(phi, 0.5, 1 / nu) + log_phi)
    dlog_log_phi = ((-1.5 / phi) + (1 / nu) / (np.power(phi, 2))) * phi + 1
    dnu_dlognu = nu
    dlog_log_nu = (-0.5 / nu + 1 / (np.power(nu, 2) * phi)) * dnu_dlognu

    return [log_p_log_phi, dlog_log_phi, dlog_log_nu]
