import numpy as np


def log_post_normal(x, mu):
    """A function that provides two outputs:
    1) Value of the log-posterior of the model at theta
    2) The gradients of the log posterior with
    respect to the parameters of the model
    """
    T, K = x.shape
    const = -0.5 * K * np.log(2 * np.pi)
    xc = x - mu.transpose()

    term1 = const - 0.5 * ((xc * xc).sum(axis=1))
    logp = sum(term1.transpose())
    dmu = np.reshape((xc.sum(axis=0)), (K, 1), order="F")

    return logp, dmu
