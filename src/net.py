from src.models import custom_feedforward
from src.utils import set_model_weights
from keras.losses import BinaryCrossentropy
from keras.layers import Activation
from keras.optimizers import Adadelta, Adam
from keras.wrappers.scikit_learn import KerasClassifier


class Net(object):
    """Standard sklearn compatible neural net"""

    def __init__(
        self,
        in_dim,
        hidden_layers,
        epochs,
        batch_size,
        verbose,
        rho,
        epsilon,
        optimizer,
        learning_rate,
        beta_1,
        beta_2,
        init_values,
    ):

        self.lr = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.init_values = init_values
        if optimizer == "ADADELTA":
            # as per original paper lr == 1.0
            self.optimizer = Adadelta(learning_rate=1.0, rho=rho, epsilon=epsilon)

        elif optimizer == "ADAM":
            self.optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
            )

        self.epochs = epochs
        self.in_dim = in_dim
        self.hidden_layers = hidden_layers
        self.model = self.get_model(epochs, batch_size, verbose)

    def create_model(self):
        """Used by KerasClassifier to build the model"""
        model = custom_feedforward(in_dim=self.in_dim, hidden_layers=self.hidden_layers)
        model.add(Activation("sigmoid"))
        model = set_model_weights(model, self.init_values["theta"])
        model.compile(
            loss=BinaryCrossentropy(from_logits=False, reduction="sum_over_batch_size"),
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )
        return model

    def get_model(self, epochs, batch_size, verbose):
        """Returns a scikit-learn compatible classifier instance"""
        return KerasClassifier(
            build_fn=self.create_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
