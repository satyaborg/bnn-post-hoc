import os
import sys
import random
import logging
import numpy as np
import pandas as pd

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

from src.vbutils.yeo_johnson import tYJi, eta2tau

import tensorflow as tf
import tensorflow.keras.backend as K

logger = logging.getLogger("train_log")


def set_rand_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_tf_loglevel(level):
    """Sets log level for TF"""
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    logging.getLogger("tensorflow").setLevel(level)


def get_logger(fname):
    """Logging function"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"logs/{fname}.log",  # tensorboard format
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger("train_log")
    # logs to console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_total_params(model):
    return np.sum([K.count_params(w) for w in model.trainable_weights])


def init_params(model, p):
    """Initializes parameters of a network"""
    q = np.sum([K.count_params(w) for w in model.trainable_weights])
    z = np.random.randn(p, 1)
    eps = np.random.randn(q, 1)
    VBtransfobj = None
    if VBtransfobj == None:
        B = np.zeros((q, p)) + 0.001
        for i in range(p):
            for j in range(i + 1):
                B[j, i] = 0
        mu = np.ones((q, 1)) * 0.001
        d = np.ones((q, 1)) * 0.01
        eta = np.ones((q, 1))

    tau = eta2tau(eta)
    phi = mu + B @ z + d * eps
    theta = tYJi(phi, eta)
    return dict(z=z, eps=eps, B=B, mu=mu, d=d, eta=eta, tau=tau, phi=phi, theta=theta)


def set_model_weights(model, theta):
    """Sets new weights (theta) to the models"""
    for i, layer in enumerate(model.layers):
        old_weights = layer.get_weights()
        new_weights = []
        for w in old_weights:
            n_params = np.prod(w.shape)
            new_weights.append(
                theta[
                    :n_params,
                ].reshape(w.shape)
            )
            theta = np.delete(theta, np.s_[:n_params], axis=0)
        model.layers[i].set_weights(new_weights)

    return model


def gamma_prior(eta):
    """Gamma prior"""
    alpha, beta = 1, 1
    _lambda = np.exp(eta)
    llgam = (
        alpha * log(beta) - log(gamma(alpha)) + (alpha - 1) * eta - beta * _lambda + eta
    )
    dllgam = (alpha - 1) - beta * _lambda + 1
    return llgam, dllgam


def cross_entropy(y_hat, y):
    y_hat = np.clip(y_hat, 1e-16, 1 - 1e-16)
    cs = -y * np.log(y_hat) - (1.0 - y) * np.log(1.0 - y_hat)
    return np.mean(cs)


def brier_score(y_hat, y):
    y_hat = np.clip(y_hat, 0.0, 1.0)
    return np.power(y_hat - y, 2.0).mean()


def plot_losses(vb_test_loss, std_test_loss):
    plt.figure(figsize=(11, 7))
    df_test_loss = pd.DataFrame(
        zip(vb_test_loss, std_test_loss, range(len(vb_test_loss))),
        columns=["VB loss", "NN loss", "fold"],
    )
    df_test_loss.plot(x="fold", y=["VB loss", "NN loss"], kind="bar")
    plt.legend(["VB", "NN"])
    plt.xlabel("Folds")
    plt.ylabel("Loss")
    mean_vb = np.mean(vb_test_loss)
    mean_nn = np.mean(std_test_loss)
    plt.axhline(y=mean_vb, color="blue", linestyle="--")
    plt.axhline(y=mean_nn, color="orange", linestyle="--")
    plt.title(f"Mean losses: VB={mean_vb:.4f}; NN={mean_nn:.4f}", fontsize=12)
    plt.tight_layout()
    return plt


def plot_nn_loss(loss, val_loss):
    fig = plt.figure(figsize=(11, 7))
    plt.plot(loss, label="train")
    plt.plot(val_loss, label="validation")
    # track min val_loss
    ymin = np.min(val_loss) if np.min(val_loss) < np.min(loss) else np.min(loss)
    ymax = np.max(val_loss) if np.max(val_loss) > np.max(loss) else np.max(loss)
    plt.vlines(
        np.argmin(val_loss), ymin=ymin, ymax=ymax, color="red", linestyle="dashed"
    )
    plt.legend()
    fig.suptitle(
        f"NN log-loss [min loss/epoch]=[{np.min(val_loss):.4f}/{np.argmin(val_loss)+1}]"
    )
    plt.tight_layout()
    return fig


def plot_vb_loss(loss, val_loss):
    fig = plt.figure(figsize=(11, 7))
    plt.plot(loss, label="train")
    plt.plot(val_loss, label="validation")
    # track min val_loss
    ymin = np.min(val_loss) if np.min(val_loss) < np.min(loss) else np.min(loss)
    ymax = np.max(val_loss) if np.max(val_loss) > np.max(loss) else np.max(loss)
    plt.vlines(
        np.argmin(val_loss), ymin=ymin, ymax=ymax, color="red", linestyle="dashed"
    )
    plt.legend()
    fig.suptitle(
        f"VB log-loss [min loss/epoch]=[{np.min(val_loss):.4f}/{np.argmin(val_loss)+1}]"
    )
    plt.tight_layout()
    return fig


def plot_metrics(elbo, trace):
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(gs[0, :])  # row 0, col 0
    ax1.title.set_text("ELBO")
    ax1.plot(elbo)
    ax3 = fig.add_subplot(gs[1, :])  # row 1, span all columns
    ax3.title.set_text("Parameter trace")
    ax3.plot(trace)
    plt.tight_layout()
    plt.tight_layout()
    return fig


def save_plot_data(fpath, fname, do_plot=False, **kwargs):
    """Saves plots and numpy arrays"""
    # save plot
    if do_plot:
        plt.savefig(fpath / (fname + ".pdf"))
    # save numpy arrays
    np.savez(fpath / (fname + ".npz"), **kwargs)


def plot_theta(x, i):
    fig = plt.figure(figsize=(11, 7))
    plt.hist(x[i, :])
    fig.suptitle(f"Distribution of weights for parameter {i+1}")
    plt.tight_layout()
    return fig


def plot_weights(vb, nn, bins=10):
    fig = plt.figure(figsize=(11, 7))
    x = np.concatenate([w.flatten() for w in vb.get_weights()])
    y = np.concatenate([w.flatten() for w in nn.get_weights()])
    plt.hist(x, bins, alpha=0.5, label="VB")
    plt.hist(y, bins, alpha=0.5, label="NN")
    plt.legend(loc="upper right")
    fig.suptitle(f"Distribution of trained weights")
    plt.tight_layout()
    return fig


def pca(x_test, seed):
    if x_test.shape[1] > 2:
        # dimensionality reduction
        pca_obj = PCA(n_components=2, random_state=seed)
        pca = pca_obj.fit_transform(x_test)
        df = pd.DataFrame(pca, columns=["x1", "x2"])
    else:
        df = pd.DataFrame(list(x_test), columns=["x1", "x2"])

    return df


def plot_data(x_test, y_test, fig, ax, plot_title, seed):
    """Plots predicted classes with different hues"""
    df = pca(x_test, seed)
    df["label"] = y_test
    eps = 1e-10
    df["intensity"] = df.label
    df.loc[df["label"] == 0, "intensity"] = eps
    hues = (
        np.array(
            [
                [0, 0, 255],
                [0, 50, 204],
                [0, 101, 153],
                [0, 153, 101],
                [0, 204, 50],
                [0, 255, 0],
                [51, 203, 0],
                [102, 152, 0],
                [153, 101, 0],
                [203, 51, 0],
                [255, 0, 0],
            ]
        )
        / 255
    )
    hues = [tuple(hue) for hue in list(hues)]
    nhues = len(hues)
    hex_values = list(map(lambda x: to_hex(x, keep_alpha=False), hues))
    # assign hues
    bins = list(np.linspace(0.05, 0.95, num=nhues - 2 + 1))
    bins.append(1)
    bins.insert(0, 0)
    cmap = LinearSegmentedColormap.from_list("class_cmap", hues, N=len(bins))
    df["hex"] = pd.cut(df["intensity"], bins, labels=hex_values)
    norm = plt.Normalize(0, 1)
    plot = ax.scatter(
        x=df.x1,
        y=df.x2,
        c=df.intensity,
        # marker='^',
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolors="w",  # set it to 'none' for remove marker boundary or 'w'
        linewidths=0.45
        # alpha=0.25
    )
    ax.set_title(plot_title, fontsize=18)
    return plot, fig


def create_folder(path):
    # create results folder
    if not os.path.exists(path):
        os.mkdir(path)


def plot_val_predictions(x_test, y_test, y_pred, seed):
    fig = plt.figure(figsize=(16, 12))  # tight_layout=True
    ax = plt.subplot(1, 2, 1)
    plot, fig = plot_data(
        x_test, y_test, fig, ax, plot_title="True Classification", seed=seed
    )
    # Predicted plot
    ax = plt.subplot(1, 2, 2)
    plot, fig = plot_data(
        x_test, y_pred, fig, ax, plot_title="Fitted Classification", seed=seed
    )
    # add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [x0, y0, width, height]
    cbar = fig.colorbar(plot, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # fig.tight_layout()
    return fig


def plot_predictions(name, methods, mean_probas, losses, briers, x_test, y_test, seed):
    """Plots predictions on test set"""
    eps = 1e-10
    hues = (
        np.array(
            [
                [0, 0, 255],
                [0, 50, 204],
                [0, 101, 153],
                [0, 153, 101],
                [0, 204, 50],
                [0, 255, 0],
                [51, 203, 0],
                [102, 152, 0],
                [153, 101, 0],
                [203, 51, 0],
                [255, 0, 0],
            ]
        )
        / 255
    )
    hues = [tuple(hue) for hue in list(hues)]
    nhues = len(hues)
    # assign hues
    bins = list(np.linspace(0.05, 0.95, num=nhues - 2 + 1))
    bins.append(1)
    bins.insert(0, 0)
    cmap = LinearSegmentedColormap.from_list("class_cmap", hues, N=len(bins))
    df = pca(x_test, seed)  # get 2 principle components
    x1, x2 = df.x1, df.x2
    norm = plt.Normalize(0, 1)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    ax1.title.set_text("True")
    ax1.title.set_size(14)
    labels = y_test.squeeze()
    labels[labels == 0] = eps
    # print(df.intensity)
    ax1.scatter(
        x=x1,
        y=x2,
        c=labels,
        # marker='^',
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolors="w",  # set it to 'none' for remove marker boundary or 'w'
        linewidths=0.45
        # alpha=0.25
    )
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.title.set_text("Variational Bayes")
    ax2.title.set_size(14)
    labels = mean_probas["var_bayes"].squeeze()
    labels[labels == 0] = eps
    ax2.scatter(
        x=x1,
        y=x2,
        c=labels,
        # marker='^',
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolors="w",  # set it to 'none' for remove marker boundary or 'w'
        linewidths=0.45
        # alpha=0.25
    )
    # place a text box in upper left in axes coords
    ax2.text(
        0.05,
        0.95,
        f"log-loss={losses['var_bayes']:.6f}\nbrier score={briers['var_bayes']:.6f}",
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.title.set_text("None")
    ax3.title.set_size(14)
    labels = mean_probas[None].squeeze()
    labels[labels == 0] = eps
    ax3.scatter(
        x=x1,
        y=x2,
        c=labels,
        # marker='^',
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolors="w",  # set it to 'none' for remove marker boundary or 'w'
        linewidths=0.45
        # alpha=0.25
    )
    # place a text box in upper left in axes coords
    ax3.text(
        0.05,
        0.95,
        f"log-loss={losses[None]:.6f}\nbrier score={briers[None]:.6f}",
        transform=ax3.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.title.set_text("Beta")
    ax4.title.set_size(14)
    labels = mean_probas["beta"].squeeze()
    labels[labels == 0] = eps
    ax4.scatter(
        x=x1,
        y=x2,
        c=labels,
        # marker='^',
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolors="w",  # set it to 'none' for remove marker boundary or 'w'
        linewidths=0.45
        # alpha=0.25
    )
    # place a text box in upper left in axes coords
    ax4.text(
        0.05,
        0.95,
        f"log-loss={losses['beta']:.6f}\nbrier score={briers['beta']:.6f}",
        transform=ax4.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    fig.suptitle(f"Test set predictions - {name}", size=18)
    fig.tight_layout()
    return fig


def plot_calibration_curve(mean_probas, losses, X_test, y_test, methods, name):
    """Plot calibration curve for est w/o and with calibration.
    # Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
    # License: BSD Style.
    """
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for method in methods:
        if method in ["beta_am", "beta_ab"]:
            continue

        prob_pos = mean_probas[method]
        clf_score = losses[method]
        if method == None or method == "None":
            label = "Uncalibrated"

        elif method == "var_bayes":
            label = "Var Bayes"

        elif method == "sigmoid":
            label = "Logistic"

        else:
            label = method[0].upper() + method[1:]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )
        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label="%s (%1.3f)" % (label, clf_score),
        )
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=method, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f"Reliability Plot")
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    return fig


def get_brier_scores(
    name, model, cal_method, x_train, y_train, x_test, y_test, cv, bins
):

    """[summary]

    Return

    prob_true: ndarray of shape (n_bins,) or smaller
    The proportion of samples whose class is the positive class, in each bin (fraction of positives).

    prob_pred: ndarray of shape (n_bins,) or smaller
    The mean predicted probability in each bin.
    """
    calibrated_model = CalibratedClassifierCV(model, method=cal_method, cv=cv)
    calibrated_model.fit(x_train, y_train)
    # predict probabilities
    y_pred_cal = calibrated_model.predict_proba(x_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred_cal, n_bins=bins)
    plt.plot(prob_pred, prob_true, marker=".", color="red", label=f"calibrated {name}")
    brier_score = brier_score_loss(y_test, y_pred_cal)
    logger.info("Brier Score [calibrated {name}]: {:.6f}".format(brier_score))
    return brier_score


def expected_calibration_error(y_true, y_pred, num_bins=10):
    """Computes the ECE
    Adapted from: https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
    """
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)
    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)
    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]
