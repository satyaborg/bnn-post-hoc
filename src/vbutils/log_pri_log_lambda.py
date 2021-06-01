import numpy as np
from src.vbutils.igamma_logpdf import igamma_logpdf


def log_pri_log_lambda(_lambda, log_lambda, psi_var):

    log_p_log_lambda = igamma_logpdf(_lambda, 0.5, 1 / psi_var) + log_lambda
    dlog_log_lambda = (
        (-1.5 / _lambda) + (1 / psi_var) / np.power(_lambda, 2)
    ) * _lambda + 1
    dpsi_var_dlogpsi_var = psi_var
    dlog_log_psi_var = (
        -0.5 / psi_var + 1 / (np.power(psi_var, 2) @ _lambda)
    ) * dpsi_var_dlogpsi_var

    return [log_p_log_lambda, dlog_log_lambda, dlog_log_psi_var]
