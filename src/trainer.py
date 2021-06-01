# general
import sys
import time
import copy
import logging
import numpy as np
import numpy.matlib
from tqdm import trange

# custom modules
from src.utils import (
    create_folder,
    plot_metrics,
    plot_vb_loss,
)

# VB functions
from src.vbutils.yeo_johnson import tau2eta, dtheta_dtau
from src.vbutils.grad_theta_logq_func import grad_theta_logq
from src.vbutils.dtheta_dmu_func import dtheta_dmu
from src.vbutils.dtheta_dBDelta_func import dtheta_dBDelta
from src.vbutils.yeo_johnson import tau2eta, eta2tau, tYJi, dtYJ_dtheta
from src.vbutils.general_functions import vec, vec2mat
from src.vbutils.logmvnpdf_funcs import logmvnpdf2

# tensorflow/keras
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import BinaryCrossentropy
from src.models import custom_feedforward

tf.keras.backend.clear_session()  # clear others models from memory

logger = logging.getLogger("train_log")

logger.info("TF version : {}".format(tf.__version__))
logger.info("GPU Available: {}".format(str(tf.config.list_physical_devices("GPU"))))


class Trainer(object):
    """
    Objective: To approximate p(θ|y) using variational inference

    1. Density of interest: Params of the NN conditional on the data
        p(θ|y)∝p(y|θ)p(θ)
        p(θ) = prior density

    2. Member of some parametric family, density: qλ(θ),
    Used for approximating p(θ|y)

    3. λ = Vector of variational parameters

    4. Variational inference: Optimization problem
        Minimizing the KL divergence between qλ(θ) and p(θ|y) w.r.t λ

    4. KL divergence (not symmetric):
    KL(qλ(θ)||p(θ|y)) = logp(y)−L(λ)

    5. L(λ) = ELBO (Evidence lower bound)
            = Eqλ[logh(θ)−logqλ(θ)]

        -- eq (I)
        Where, h(θ) = p(y|θ)p(θ)

    6. Since logp(y) does not depend on λ,
        Minimizing (4) is equivalent to maximizing L(λ)

    7. SGA for optimizing
        Update rule:
        λ(i+1):=λ(i)+ρi◦̂∇λL(λ(i))

        where,
        ρi = (ρi1,...,ρim) = Vector of step sizes
        ∇λL(λ(i)) = Unbiased estimate of the gradient of L(λ) atλ=λ(i)

    8. ADAM is used for to set the step sizes (ρi) -> rho_i

    9. To obtain ∇λL(λ(i)),
    These can be obtained by evaluating the derivative of the argument of the
    expectation in Equation (I), at multiple draws from qλ, and then averaging
    over all the derivatives

    10. Reparametrization trick is employed to reduce variance of the estimate
    of ∇λL(λ(i))

    11. (λ) =Efε[logh(k(ε,λ))−logqλ(k(ε,λ))]
    """

    def __init__(self, in_dim, init_values, **config):
        self.debug = config.get("debug")
        self.seed = config.get("seed")
        self.in_dim = in_dim
        self.out_dim = config.get("out_dim")
        self.hidden_layers = config.get("hidden_layers")
        self.paths = config.get("paths")
        self.batch_size = config.get("batch_size")
        self.n_samples = None
        self.valid_pct = config.get("valid_pct")
        self.model_arch = config.get("model_arch")
        self.lr = config.get("lr")
        self.filename = config.get("filename")
        self.p = config.get("p")
        self.transf = config.get("transf")
        self.epochs = config.get("vb_epochs")
        self.randomize_minibatch = config.get("randomize_minibatch")
        self.train_dataset = None
        self.test_dataset = None
        self.y_test = None
        self.factor = None
        self.model = None
        self.optimizer = config.get("optimizer")
        self.epsilon = config.get("epsilon")  # 1e-07
        self.rho = config.get("rho")
        self.init_values = copy.deepcopy(init_values)  # w/o deepcopy values mutate
        self.learning_rate = config.get("vb_lr")
        self.beta_1 = config.get("beta_1")
        self.beta_2 = config.get("beta_2")
        self.bce = BinaryCrossentropy(
            from_logits=False, reduction="sum_over_batch_size"
        )
        self.monitor_loss = config.get("monitor_loss")
        self.prior = config.get("prior")
        self.mode = config.get("mode")

    def VBtransf(
        self,
        valid_set,
        NNpar,
        p,
        Transf,
        fpath,
        name,
        VBtransfobj=None,
    ):
        """VBtransf function

        Args:
            q ([type]): [description]
            p ([type]): [description]
            Transf ([type]): [description]
            niter ([type]): [description]
            VBtransfobj ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        q = NNpar
        if VBtransfobj == None:
            # the initial values
            B = self.init_values["B"]
            mu = self.init_values["mu"]
            d = self.init_values["d"]
            eta = self.init_values["eta"]

        else:
            B = VBtransfobj[0]
            mu = VBtransfobj[1]
            d = VBtransfobj[2]
            eta = VBtransfobj[3]
            Transf = VBtransfobj[4]

        # mu
        Edelta2_mu = np.zeros(mu.shape)
        Eg2_mu = np.zeros(mu.shape)
        # B
        Edelta2_B = np.zeros((B.shape[0] * B.shape[1], 1))
        Eg2_B = np.zeros((B.shape[0] * B.shape[1], 1))
        # d
        Edelta2_d = np.zeros(d.shape)
        Eg2_d = np.zeros(d.shape)
        # tau
        Edelta2_tau = np.zeros(eta.shape)
        Eg2_tau = np.zeros(eta.shape)
        # ADADELTA
        ADA = [
            self.rho,
            self.epsilon,
            Edelta2_mu,
            Eg2_mu,
            Edelta2_B,
            Eg2_B,
            Edelta2_d,
            Eg2_d,
            Edelta2_tau,
            Eg2_tau,
        ]
        # ADAM ################################################################
        # initialize first and second moments
        M = dict(
            mu=np.zeros(mu.shape),
            B=np.zeros((B.shape[0] * B.shape[1], 1)),
            d=np.zeros(d.shape),
            tau=np.zeros(eta.shape),
        )
        V = dict(
            mu=np.zeros(mu.shape),
            B=np.zeros((B.shape[0] * B.shape[1], 1)),
            d=np.zeros(d.shape),
            tau=np.zeros(eta.shape),
        )
        StoreLB = []
        temp = []
        StoreTime = []
        val_log_loss = []
        train_log_loss = []
        mu_sum_epoch = []
        d_sum_epoch = []
        B_sum_epoch = []
        eta_sum_epoch = []
        t = time.time()
        n_epochs = 5
        for epoch in trange(self.epochs):
            for step, (x, y) in enumerate(self.train_dataset):
                if self.optimizer == "ADADELTA":
                    LowerB, B, mu, d, eta, ADA = self.VB_step(
                        x, y, B, mu, d, eta, ADA, p, epoch, step, Transf
                    )

                elif self.optimizer == "ADAM":
                    LowerB, B, mu, d, eta, M, V = self.VB_step_w_adam(
                        x, y, B, mu, d, eta, M, V, p, epoch, step, Transf
                    )
                if any(np.isnan(eta)):
                    sys.exit(
                        "The number of inputs should be equal to the number of tranformation parameters"
                    )

                StoreLB.append(LowerB.squeeze().item())
                temp.append(mu[mu.shape[0] - 1].squeeze().item())
                StoreTime.append(time.time() - t)

            # end of epoch ****************************************************
            mu_sum_epoch.append(mu)
            d_sum_epoch.append(d)
            B_sum_epoch.append(B)
            eta_sum_epoch.append(eta)
            if self.monitor_loss and self.mode == "validation":
                F_mu = np.mean(mu_sum_epoch[-n_epochs:], axis=0)
                F_d = np.mean(d_sum_epoch[-n_epochs:], axis=0)
                F_B = np.mean(B_sum_epoch[-n_epochs:], axis=0)
                F_eta = np.mean(eta_sum_epoch[-n_epochs:], axis=0)
                self.set_net_weights(F_mu, F_d, F_B, eta, NNpar, n=5000)
                val_log_loss.append(self.get_loss(valid_set))
                train_log_loss.append(self.get_loss(self.train_dataset))
                logger.info(
                    f"epoch: {epoch+1} - loss: {train_log_loss[-1]:.4f} - val_loss: {val_log_loss[-1]:.4f}"
                )

            # plot ELBO every 10 epochs
            if (epoch + 1) % 10 == 0:
                fig = plot_metrics(elbo=StoreLB, trace=temp)
                fig.savefig(f"{fpath}/{name}_{run}_{self.mode}_elbo_plot.pdf")
                if self.monitor_loss and self.mode == "validation":
                    fig = plot_vb_loss(loss=train_log_loss, val_loss=val_log_loss)
                    fig.savefig(f"{fpath}/{name}_{run}_{self.mode}_vb_loss.pdf")

        # training ended ######################################################
        # mean across last n_epochs
        F_mu = np.mean(mu_sum_epoch[-n_epochs:], axis=0)
        F_d = np.mean(d_sum_epoch[-n_epochs:], axis=0)
        F_B = np.mean(B_sum_epoch[-n_epochs:], axis=0)
        F_eta = np.mean(eta_sum_epoch[-n_epochs:], axis=0)
        F_LB = StoreLB
        logger.info(f"Total time elapsed: {time.time() - t:.4f} secs")
        cv_epochs = np.argmin(val_log_loss) + 1 if self.mode == "validation" else None
        return F_mu, F_d, F_B, F_eta, F_LB, cv_epochs, StoreTime

    def adam_optimizer(
        self, mu, B, d, tau, L_mu, L_B, L_d, L_tau, M, V, p, q, epoch, transf
    ):
        """ADAM optimizer
        Adapted from: https://machinelearningmastery.com/adam-optimization-from-scratch/
        """
        # mu update ###########################################################
        # m(t) = beta_1 * m(t-1) + (1 - beta_1) * g(t)
        M["mu"] = self.beta_1 * M["mu"] + (1.0 - self.beta_1) * L_mu
        # v(t) = beta_2 * v(t-1) + (1 - beta_2) * g(t)^2
        V["mu"] = self.beta_2 * V["mu"] + (1.0 - self.beta_2) * np.power(L_mu, 2)
        # mhat(t) = m(t) / (1 - beta_1(t))
        mhat = M["mu"] / (1.0 - self.beta_1 ** (epoch + 1))
        # vhat(t) = v(t) / (1 - beta_2(t))
        vhat = V["mu"] / (1.0 - self.beta_2 ** (epoch + 1))
        # parameter updates
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        mu += (self.learning_rate * mhat) / (np.sqrt(vhat) + self.epsilon)
        # B update ###########################################################
        vecL_B = np.reshape(vec(L_B), (q * p, 1))
        # m(t) = beta_1 * m(t-1) + (1 - beta_1) * g(t)
        M["B"] = self.beta_1 * M["B"] + (1.0 - self.beta_1) * vecL_B
        # v(t) = beta_2 * v(t-1) + (1 - beta_2) * g(t)^2
        V["B"] = self.beta_2 * V["B"] + (1.0 - self.beta_2) * np.power(vecL_B, 2)
        # mhat(t) = m(t) / (1 - beta_1(t))
        mhat = M["B"] / (1.0 - self.beta_1 ** (epoch + 1))
        # vhat(t) = v(t) / (1 - beta_2(t))
        vhat = V["B"] / (1.0 - self.beta_2 ** (epoch + 1))
        # parameter updates
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        change_delta_B = (self.learning_rate * mhat) / (np.sqrt(vhat) + self.epsilon)
        B += vec2mat(change_delta_B, q, p)
        # d update ###########################################################
        # m(t) = beta_1 * m(t-1) + (1 - beta_1) * g(t)
        M["d"] = self.beta_1 * M["d"] + (1.0 - self.beta_1) * L_d
        # v(t) = beta_2 * v(t-1) + (1 - beta_2) * g(t)^2
        V["d"] = self.beta_2 * V["d"] + (1.0 - self.beta_2) * np.power(L_d, 2)
        # mhat(t) = m(t) / (1 - beta_1(t))
        mhat = M["d"] / (1.0 - self.beta_1 ** (epoch + 1))
        # vhat(t) = v(t) / (1 - beta_2(t))
        vhat = V["d"] / (1.0 - self.beta_2 ** (epoch + 1))
        # parameter updates
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        d += (self.learning_rate * mhat) / (np.sqrt(vhat) + self.epsilon)
        # tau update ###########################################################
        # m(t) = beta_1 * m(t-1) + (1 - beta_1) * g(t)
        M["tau"] = self.beta_1 * M["tau"] + (1.0 - self.beta_1) * L_tau
        # v(t) = beta_2 * v(t-1) + (1 - beta_2) * g(t)^2
        V["tau"] = self.beta_2 * V["tau"] + (1.0 - self.beta_2) * np.power(L_tau, 2)
        # mhat(t) = m(t) / (1 - beta_1(t))
        mhat = M["tau"] / (1.0 - self.beta_1 ** (epoch + 1))
        # vhat(t) = v(t) / (1 - beta_2(t))
        vhat = V["tau"] / (1.0 - self.beta_2 ** (epoch + 1))
        # parameter updates
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        change_delta_tau = (self.learning_rate * mhat) / (np.sqrt(vhat) + self.epsilon)
        taustep = np.reshape(change_delta_tau, tau.shape, "F")
        tau += taustep * transf
        return mu, B, d, tau, M, V

    def VB_step_w_adam(self, x, y, B, mu, d, eta, M, V, p, epoch, step, Transf=None):
        """VB step function"""
        if Transf == None:
            Transformation = 0
            Transf = "YJ"

        else:
            Transformation = 1

        q = mu.shape[0]
        # for the first iteration use same values as NN
        if epoch == 0 and step == 0:
            z = self.init_values["z"]
            eps = self.init_values["eps"]
            tau = self.init_values["tau"]
            phi = self.init_values["phi"]
            theta = self.init_values["theta"]

        else:
            z = np.random.randn(p, 1)
            eps = np.random.randn(q, 1)
            tau = eta2tau(eta)
            phi = mu + B @ z + d * eps
            theta = tYJi(phi, eta)  # shape: (no. of params, 1)

        L_mu, L_B, L_d, L_tau, g = self.gradient_compute(
            x, y, theta, mu, B, z, d, eps, tau, phi, Transf
        )
        L_B[np.tril(np.ones((L_B.shape))) != 1] = 0
        mu, B, d, tau, M, V = self.adam_optimizer(
            mu, B, d, tau, L_mu, L_B, L_d, L_tau, M, V, p, q, epoch, Transformation
        )
        eta = tau2eta(tau)
        # lower bound
        loghtheta = g
        phi = mu + B @ z + d * eps
        theta = tYJi(phi, eta)
        dt_dtheta = dtYJ_dtheta(theta, eta)
        logphiNorm = logmvnpdf2(B, z, d, eps)
        logJacobian = sum(np.log(dt_dtheta))
        LowerB = loghtheta - logJacobian - logphiNorm
        return LowerB, B, mu, d, eta, M, V

    def VB_step(self, x, y, B, mu, d, eta, ADA, p, epoch, step, Transf=None):
        """VB step function"""
        if Transf == None:
            Transformation = 0
            Transf = "YJ"
        else:
            Transformation = 1

        rho = ADA[0]
        eps_step = ADA[1]
        oldEdelta2_mu = ADA[2]
        oldEg2_mu = ADA[3]
        oldEdelta2_B = ADA[4]
        oldEg2_B = ADA[5]
        oldEdelta2_d = ADA[6]
        oldEg2_d = ADA[7]
        oldEdelta2_tau = ADA[8]
        oldEg2_tau = ADA[9]
        q = mu.shape[0]
        # for the first iteration use same values as NN
        if epoch == 0 and step == 0:
            z = self.init_values["z"]
            eps = self.init_values["eps"]
            tau = self.init_values["tau"]
            phi = self.init_values["phi"]
            theta = self.init_values["theta"]

        else:
            z = np.random.randn(p, 1)
            eps = np.random.randn(q, 1)
            tau = eta2tau(eta)
            phi = mu + B @ z + d * eps
            theta = tYJi(phi, eta)

        # compute all gradients
        L_mu, L_B, L_d, L_tau, g = self.gradient_compute(
            x, y, theta, mu, B, z, d, eps, tau, phi, Transf
        )
        L_B[np.tril(np.ones((L_B.shape))) != 1] = 0
        # ADADELTA Optimizer ##################################################
        # "mu update"
        ADA[3] = rho * oldEg2_mu + (1 - rho) * np.power(L_mu, 2)
        Change_delta_mu = (
            np.sqrt(oldEdelta2_mu + eps_step) / np.sqrt(ADA[3] + eps_step)
        ) * L_mu
        mu = mu + Change_delta_mu
        ADA[2] = rho * oldEdelta2_mu + (1 - rho) * np.power(Change_delta_mu, 2)
        # "B update"
        vecL_B = np.reshape(vec(L_B), (q * p, 1))
        ADA[5] = rho * oldEg2_B + (1 - rho) * np.power(vecL_B, 2)
        Change_delta_B = (
            np.sqrt(oldEdelta2_B + eps_step) / np.sqrt(ADA[5] + eps_step) * vecL_B
        )
        B = B + vec2mat(Change_delta_B, q, p)
        ADA[4] = rho * oldEdelta2_B + (1 - rho) * np.power(Change_delta_B, 2)
        # "d update"
        ADA[7] = rho * oldEg2_d + (1 - rho) * np.power(L_d, 2)
        Change_delta_d = (
            np.sqrt(oldEdelta2_d + eps_step) / np.sqrt(ADA[7] + eps_step) * L_d
        )
        d = d + Change_delta_d
        ADA[6] = rho * oldEdelta2_d + (1 - rho) * np.power(Change_delta_d, 2)
        # "tau update"
        ADA[9] = rho * oldEg2_tau + (1 - rho) * np.power(L_tau, 2)
        Change_delta_tau = (
            np.sqrt(oldEdelta2_tau + eps_step) / np.sqrt(ADA[9] + eps_step) * L_tau
        )
        taustep = np.reshape(Change_delta_tau, tau.shape, "F")
        tau = tau + taustep * Transformation
        ADA[8] = rho * oldEdelta2_tau + (1 - rho) * np.power(Change_delta_tau, 2)
        eta = tau2eta(tau)
        # lower bound
        loghtheta = g
        phi = mu + B @ z + d * eps
        theta = tYJi(phi, eta)
        dt_dtheta = dtYJ_dtheta(theta, eta)
        logphiNorm = logmvnpdf2(B, z, d, eps)
        logJacobian = sum(np.log(dt_dtheta))
        LowerB = loghtheta - logJacobian - logphiNorm
        return LowerB, B, mu, d, eta, ADA

    def gradient_compute(self, x, y, theta, mu, B, z, d, eps, tau, phi, Transf="YJ"):
        """Computes and returns the gradients"""
        q, p = B.shape
        eta = tau2eta(tau)
        g, delta_logh = self.ll_net(x, y, theta)
        delta_logq = grad_theta_logq(theta, eta, mu, B, d, phi, Transf)
        L_mu = dtheta_dmu(phi, eta, theta, Transf) @ (delta_logh - delta_logq)
        dtheta_dB, dtheta_dd = dtheta_dBDelta(eta, B, z, eps, phi, theta, Transf)
        L_B = np.reshape(dtheta_dB.transpose() @ (delta_logh - delta_logq), (q, p), "F")
        L_d = dtheta_dd.transpose() @ (delta_logh - delta_logq)
        L_tau = (dtheta_dtau(phi, tau, theta, Transf).transpose()) @ (
            delta_logh - delta_logq
        )  # derivatives with respect to tau, the fisher transformation of eta
        return L_mu, L_B, L_d, L_tau, g

    def set_model_weights(self, theta):
        """Method to set new weights (theta) to the model"""
        for i, layer in enumerate(self.model.layers):
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

            self.model.layers[i].set_weights(new_weights)

    @tf.function
    def compute_jacobian(self, x, y):
        """Computes the Jacobian of errors"""
        batch_size = x.shape[0]
        with tf.GradientTape() as tape:
            f_NN = self.model(x, training=True)
            g_f_NN = tf.sigmoid(f_NN)
            g_f_NN = tf.clip_by_value(
                g_f_NN, clip_value_min=1e-16, clip_value_max=1 - 1e-16
            )
            # ll = self.factor * (tf.reduce_sum(tf.math.log(g_f_NN[tf.squeeze(y) == 1])) + tf.reduce_sum(tf.math.log(1 - g_f_NN[tf.squeeze(y) == 0])))
            ll = self.factor * self.bce(y, g_f_NN)

        # compute jacobian of errors w.r.t weights/biases (network parameters)
        dmu_dtheta = tape.jacobian(
            target=f_NN, sources=self.model.trainable_weights
        )  # note: d_fnn/d_w = -d_errors/d_w
        dmu_dtheta = tf.concat(
            [tf.reshape(w, [batch_size, -1]) for w in dmu_dtheta], axis=-1
        )
        return dmu_dtheta, g_f_NN, ll

    def ll_net(self, x, y, theta):
        """Returns log-likelihood"""
        # assign weights (theta) to the model
        self.set_model_weights(theta)
        # compute the jacobian, predictions
        dmu_dtheta, g_f_NN, ll = self.compute_jacobian(x, y)
        dmu_dtheta = dmu_dtheta.numpy()
        g_f_NN = g_f_NN.numpy()
        ll = ll.numpy()
        y = tf.squeeze(y).numpy()
        dgdmu_on_g = 1 - np.squeeze(g_f_NN)
        dgdmu_on_1mg = np.squeeze(g_f_NN)
        Tempvector = np.zeros(dgdmu_on_1mg.shape)
        Tempvector[y == 1] = dgdmu_on_g[y == 1]
        Tempvector[y == 0] = -dgdmu_on_1mg[y == 0]
        dll = np.sum(np.transpose(dmu_dtheta) * Tempvector, axis=1)
        dll = self.factor * dll
        dll = np.expand_dims(dll, axis=-1)
        return ll, dll

    @tf.function
    def forward_pass(self, x):
        f_NN = self.model(x, training=False)  # raw logits
        g_f_NN = tf.sigmoid(f_NN)  # 1./(1+tf.exp(-f_NN))
        g_f_NN = tf.clip_by_value(
            g_f_NN, clip_value_min=1e-16, clip_value_max=1 - 1e-16
        )
        return g_f_NN

    # @profile
    def get_loss(self, test_dataset):
        """Predicts probability with a trained network"""
        log_loss = []
        for x_batch_test, y_batch_test in test_dataset:
            g_f_NN = self.forward_pass(x_batch_test)
            loss = self.bce(y_batch_test, g_f_NN)
            log_loss.append(loss)

        mean_log_loss = tf.reduce_mean(log_loss)
        return mean_log_loss

    def predict_proba(self, x_test):
        g_f_NN = self.forward_pass(x_test)
        # to make it compatible with sklearn convention
        y_pred = np.zeros([x_test.shape[0], 2])
        y_pred[:, 1] = tf.squeeze(g_f_NN)  # for class 1
        y_pred[:, 0] = 1.0 - y_pred[:, 1]  # for class 0
        return y_pred

    def set_net_weights(self, F_mu, F_d, F_B, eta, NNpar, n=5000):
        mean = F_mu.squeeze()
        covar = F_B @ np.transpose(F_B) + np.diag(F_d) ** 2
        arg_phi = np.transpose(
            np.random.multivariate_normal(mean=mean, cov=covar, size=n)
        )
        arg_eta = np.matlib.repmat(eta, 1, n)
        theta = tYJi(phi=arg_phi, eta=arg_eta)
        if self.prior == "uniform":
            theta_mean = np.mean(theta, axis=1)  # [NNpar, ]
            self.set_model_weights(theta=theta_mean)

    # @profile
    def fit(self, train, valid_set, n_samples, name, run):
        """Main training function"""
        # enable/disable eager execution
        tf.config.run_functions_eagerly(self.debug)  # in order to debug tf.function
        # create result folder
        path = f"{self.paths.get('results')}/{self.filename}"
        self.model = custom_feedforward(
            in_dim=self.in_dim, hidden_layers=self.hidden_layers
        )
        self.train_dataset = train
        self.n_samples = n_samples  # no. of training examples
        self.factor = n_samples / self.batch_size
        # no of neural net params
        NNpar = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        logger.info(f"total parameters: {NNpar}")
        p = self.p  # Number of factors in covariance matrix
        Transf = None  # 'YJ'
        # note:
        # F_mu: location parameter of the VB approximation
        # F_d: diagonal parameter values of D  in represenation BB'+D^2
        # F_B: parameter values in factor matrix B of represenation BB'+D^2
        # F_eta: parameter values of the transformations used in the
        # approximation
        F_mu, F_d, F_B, eta, F_LB, cv_epochs, StoreTime = self.VBtransf(
            valid_set,
            NNpar=NNpar,
            p=p,
            Transf=Transf,
            fpath=path,
            name=name,
            VBtransfobj=None,
        )
        self.set_net_weights(F_mu, F_d, F_B, eta, NNpar, n=5000)
        return cv_epochs
