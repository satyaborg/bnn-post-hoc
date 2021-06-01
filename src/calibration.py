"""Code for the calibration methods is adapted from: 
https://github.com/betacal/aistats2017/tree/master/experiments
"""
import gc
import copy
import logging
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.net import Net  # standard neural net
from src.trainer import Trainer  # VB trainer
from src.betacal.calib.models.calibration import CalibratedClassifierCV

import keras
import tensorflow as tf
from keras.losses import BinaryCrossentropy

from src.utils import (
    brier_score,
    plot_losses,
    plot_calibration_curve,
    plot_predictions,
    expected_calibration_error,
    plot_nn_loss,
    plot_val_predictions,
    plot_weights,
)

logger = logging.getLogger("train_log")
bce = BinaryCrossentropy(from_logits=False, reduction="sum_over_batch_size")


def calibrate(classifier, x_cali, y_cali, method=None, score_type=None):
    ccv = CalibratedClassifierCV(
        base_estimator=classifier, method=method, cv="prefit", score_type=score_type
    )
    ccv.fit(x_cali, y_cali)
    return ccv


def predict(test_dataset, classifier, path, name, run, model_name, mode, seed):
    """Predicts probability with a trained network"""
    all_preds = []
    y_pred = []
    x_test = []
    y_test = []
    log_loss = []
    for x_batch_test, y_batch_test in test_dataset:
        preds = classifier.predict_proba(x_batch_test)
        y_batch_pred = preds[:, 1]  # only get class 1
        y_pred.extend(y_batch_pred)
        y_test.extend(y_batch_test.numpy())
        x_test.extend(x_batch_test.numpy())
        log_loss.append(bce(y_batch_test, y_batch_pred))
        all_preds.extend(preds)

    y_test, x_test = np.array(y_test), np.array(x_test)
    fig = plot_val_predictions(x_test, y_test, y_pred, seed)
    fig.savefig(
        f"{path}/{name}_{run}_{mode}_{model_name}_test_class_plot.pdf",
        bbox_inches="tight",
    )
    return np.array(y_pred), tf.reduce_mean(log_loss).numpy(), np.array(all_preds)


def calibration(
    init_values,
    methods,
    dataset,
    x_train,
    y_train,
    x_test,
    y_test,
    score_type=None,
    **kwargs,
):

    run = 1  # dummy
    seed = kwargs.get("seed")
    keras.backend.clear_session()
    vb_test_loss = []
    nn_test_loss = []
    vb_test_loss_val = []
    nn_test_loss_val = []
    vb_epochs = copy.deepcopy(kwargs.get("vb_epochs"))
    nn_epochs = copy.deepcopy(kwargs.get("nn_epochs"))
    name = dataset.name
    path = f"{kwargs['paths'].get('results')}/{kwargs.get('filename')}"
    test_probas = {method: np.zeros(np.alen(y_test)) for method in methods}
    all_probas = {}
    # split to validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=kwargs.get("seed"),
    )
    # split to calibration set
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=kwargs.get("seed"),
    )
    # standardize with training stats
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    x_cal = scaler.transform(x_cal)
    # prepare tf datasets
    train_set = dataset.prepare_data(x_train, y_train, training=True)
    valid_set = dataset.prepare_data(x_val, y_val, training=False)
    test_set = dataset.prepare_data(x_test, y_test, training=False)
    n_samples, in_dim = x_train.shape
    logger.info("Sanity checks: ")
    logger.info(
        f"Dimensions: [ train / test / validation / calibration ]=[ {x_train.shape} / {x_test.shape} / {x_val.shape} / {x_cal.shape} ]"
    )
    logger.info(
        f"Stats [ train / test / validation / calibration ]: mean=[ {round(x_train.mean())} / {round(x_test.mean())} / {round(x_val.mean())} / {round(x_cal.mean())} ], std=[ {round(x_train.std())} / {round(x_test.std())} / {round(x_val.std())} / {round(x_cal.std())} ]"
    )
    logger.info(
        f"Class distribution: [ train / test / validation / calibration ]=[ {round(len(y_train[y_train==1])/len(y_train)*100)} / {round(len(y_test[y_test==1])/len(y_test)*100)} / {round(len(y_val[y_val==1])/len(y_val)*100)} / {round(len(y_cal[y_cal==1])/len(y_cal)*100)} ]"
    )
    for mode in ["validation", "training"]:
        keras.backend.clear_session()
        logger.info(f"==> {mode} <==")
        kwargs["mode"] = mode
        if mode == "validation":
            kwargs["vb_epochs"] = vb_epochs
            kwargs["nn_epochs"] = nn_epochs

        elif mode == "training":
            kwargs["vb_epochs"] = cv_vb_epochs  # check here!
            kwargs["nn_epochs"] = cv_nn_epochs

        # NN ##############################################################
        logger.info(f'Training NN with {kwargs["nn_epochs"]} epochs ..')
        net = Net(
            in_dim=in_dim,
            hidden_layers=kwargs.get("hidden_layers"),
            epochs=kwargs.get("nn_epochs"),
            batch_size=kwargs.get("batch_size"),
            verbose=kwargs.get("verbose"),
            rho=kwargs.get("rho"),
            epsilon=kwargs.get("epsilon"),
            optimizer=kwargs.get("optimizer"),
            learning_rate=kwargs.get("nn_lr"),
            beta_1=kwargs.get("beta_1"),
            beta_2=kwargs.get("beta_2"),
            init_values=init_values,
        )
        classifier = net.model
        history = classifier.fit(x=x_train, y=y_train, validation_data=(x_val, y_val))
        cv_nn_epochs = np.argmin(history.history["val_loss"]) + 1
        fig = plot_nn_loss(history.history["loss"], history.history["val_loss"])
        fig.savefig(f"{path}/{name}_{run}_{mode}_nn_loss.pdf")
        y_pred_nn, nn_loss, y_preds_nn = predict(
            test_set, classifier, path, name, run, "nn", mode, seed
        )
        ece_nn = expected_calibration_error(y_test, y_preds_nn)
        logger.info(f"NN ECE [{mode}]: { ece_nn }")
        logger.info(f"Test log-loss [{mode}]: { nn_loss }")
        logger.info(
            f" Probabilities [ min / max ] : [ {y_pred_nn.min()} / {y_pred_nn.max()} ]"
        )
        # VB ##############################################################
        logger.info(f'Training VB with {kwargs["vb_epochs"]} epochs ..')
        # initialize
        vbn_classifier = Trainer(in_dim=in_dim, init_values=init_values, **kwargs)
        cv_vb_epochs = vbn_classifier.fit(train_set, valid_set, n_samples, name, path)
        y_pred_vb, vb_loss, y_preds_vb = predict(
            test_set, vbn_classifier, path, name, run, "vb", mode, seed
        )
        ece_vb = expected_calibration_error(y_test, y_preds_vb)
        logger.info(f"VB ECE [{mode}]: { ece_vb }")
        logger.info(f"Test log-loss [{mode}]: { vb_loss }")
        logger.info(
            f" Probabilities [ min / max ] : [ {y_pred_vb.min()} / {y_pred_vb.max()} ]"
        )
        #######################################################################
        # store the predictions
        if mode == "training":
            test_probas["var_bayes"] = y_pred_vb
            all_probas["var_bayes"] = y_preds_vb
            test_probas[None] = y_pred_nn
            all_probas[None] = y_preds_nn
            for method in methods:
                if method != "var_bayes" and method != None:
                    if kwargs.get("verbose"):
                        print(
                            "Calibrating with " + "none" if method is None else method
                        )
                    ccv = calibrate(
                        classifier, x_cal, y_cal, method=method, score_type=score_type
                    )
                    probas = ccv.predict_proba(x_test)
                    test_probas[method] = probas[:, 1]
                    all_probas[method] = probas

            nn_test_loss.append(nn_loss)
            vb_test_loss.append(vb_loss)
            fig = plot_losses(vb_test_loss, nn_test_loss)
            fig.savefig(f"{path}/{name}_{run}_{mode}_test_loss.pdf")
            # plot weights
            fig = plot_weights(vb=vbn_classifier.model, nn=classifier.model)
            fig.savefig(
                f"{path}/{name}_{run}_weight_distribution.pdf",
                bbox_inches="tight",
            )

        else:
            nn_test_loss_val.append(nn_loss)
            vb_test_loss_val.append(vb_loss)

            fig = plot_losses(vb_test_loss_val, nn_test_loss_val)
            fig.savefig(f"{path}/{name}_{run}_{mode}_test_loss.pdf")

        del vbn_classifier
        del classifier
        gc.collect()
        logger.info("=====================================================")

    losses = {method: bce(y_test, test_probas[method]).numpy() for method in methods}
    accs = {
        method: np.mean((test_probas[method] >= 0.5) == y_test) for method in methods
    }
    briers = {method: brier_score(test_probas[method], y_test) for method in methods}
    eces = {
        method: expected_calibration_error(y_test, all_probas[method])
        for method in methods
    }
    fig = plot_predictions(
        name, methods, test_probas, losses, briers, x_test, y_test, seed
    )
    fig.savefig(f"{path}/{name}_{run}_test_class_plot.pdf", bbox_inches="tight")
    fig = plot_calibration_curve(test_probas, losses, x_test, y_test, methods, name)
    fig.savefig(f"{path}/{name}_{run}_reliability_curve.pdf", bbox_inches="tight")
    return (
        accs,
        losses,
        briers,
        eces,
        test_probas,
        np.concatenate((x_test, np.expand_dims(y_test, axis=1)), axis=1),
    )
