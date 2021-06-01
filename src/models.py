# coding: utf-8
from keras.models import Sequential
from keras.layers import Dense


def custom_feedforward(in_dim: int, hidden_layers: list = [], out_dim: int = 1):
    """A fully connected feedforward network. Creates a custom model based on
    hidden_layers.
    """
    activation = "relu"
    layers = []
    if len(hidden_layers) > 1:
        # add input layer
        layers.append(
            Dense(
                units=hidden_layers[0],
                input_dim=in_dim,
                name="input",
                activation=activation,
            )
        )
        # add hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(
                Dense(units=hidden_layers[i], name=f"hidden_{i}", activation=activation)
            )

        # add output layers
        layers.append(Dense(units=out_dim, name="output"))

    return Sequential(layers)
