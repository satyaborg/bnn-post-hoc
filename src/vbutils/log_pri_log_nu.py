import numpy as np
from src.vbutils.igamma_logpdf import igamma_logpdf


def log_pri_log_nu(nu, log_nu):
    log_p_log_nu = np.sum(igamma_logpdf(nu, 0.5, 1) + log_nu)
    dlog_log_nu = ((-1.5 / nu) + 1 / np.power(nu, 2)) * nu + 1
    return [log_p_log_nu, dlog_log_nu]
