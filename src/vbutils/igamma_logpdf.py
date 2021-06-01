import numpy as np
from scipy.special import gamma


def igamma_logpdf(x, alpha, beta):
    logpdf = (
        alpha * np.log(beta)
        - np.log(gamma(alpha))
        + (-alpha - 1) * np.log(x)
        - beta / x
    )
    return logpdf
