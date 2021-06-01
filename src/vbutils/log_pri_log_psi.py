import numpy as np
from src.vbutils.igamma_logpdf import igamma_logpdf


def log_pri_log_psi(psi_var, log_psi):
    log_p_log_psi = np.sum(igamma_logpdf(psi_var, 0.5, 1) + log_psi)
    dlog_log_psi = ((-1.5 / psi_var) + 1 / np.power(psi_var, 2)) * psi_var + 1
    return [log_p_log_psi, dlog_log_psi]
