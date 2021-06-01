import scipy.sparse as sp
from src.vbutils.yeo_johnson import dtheta_dphi


def dtheta_dmu(phi, eta, theta, Transf_type="YJ"):
    dthetadphi = dtheta_dphi(phi, eta)
    dthetadmu = sp.diags(dthetadphi.transpose(), [0])
    return dthetadmu
