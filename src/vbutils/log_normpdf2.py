import numpy as np


def log_normpdf2(x, mu, sigma):
    l_pdf = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(np.power(sigma, 2))
        - 0.5 * (np.power(x - mu, 2)) / (np.power(sigma, 2))
    )
    return l_pdf
